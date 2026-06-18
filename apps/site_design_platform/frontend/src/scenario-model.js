function deepCopy(obj) { return JSON.parse(JSON.stringify(obj)); }

function lonLatToMeters(lon, lat, lat0) {
  const x = lon * 111320 * Math.cos((lat0 * Math.PI) / 180);
  const y = lat * 111320;
  return [x, y];
}

function ringAreaSqm(ring) {
  if (!ring || ring.length < 4) return 0;
  const lat0 = ring.reduce((a, c) => a + c[1], 0) / ring.length;
  let s = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const [x1, y1] = lonLatToMeters(ring[i][0], ring[i][1], lat0);
    const [x2, y2] = lonLatToMeters(ring[i + 1][0], ring[i + 1][1], lat0);
    s += (x1 * y2) - (x2 * y1);
  }
  return Math.abs(s) / 2;
}

function polygonAreaSqm(g) {
  if (!g) return 0;
  if (g.type === "Polygon") return ringAreaSqm(g.coordinates[0] || []);
  if (g.type === "MultiPolygon") return g.coordinates.reduce((a, p) => a + ringAreaSqm((p[0] || [])), 0);
  return 0;
}

function centroidPolygon(ring) {
  let x = 0; let y = 0; let n = 0;
  ring.slice(0, -1).forEach((c) => { x += c[0]; y += c[1]; n += 1; });
  return n ? [x / n, y / n] : ring[0] || [0, 0];
}

function rotatePoint(c, o, a) {
  const dx = c[0] - o[0];
  const dy = c[1] - o[1];
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  return [o[0] + dx * cos - dy * sin, o[1] + dx * sin + dy * cos];
}

function rotateGeometry(g, angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  if (g.type === "Polygon") {
    const ring = g.coordinates[0];
    const o = centroidPolygon(ring);
    g.coordinates[0] = ring.map((c) => rotatePoint(c, o, a));
  }
}

function translateGeometry(g, dx, dy) {
  const m = (c) => [c[0] + dx, c[1] + dy];
  if (g.type === "Polygon") g.coordinates = g.coordinates.map((r) => r.map(m));
  if (g.type === "MultiPolygon") g.coordinates = g.coordinates.map((p) => p.map((r) => r.map(m)));
}

function upsertClosedRing(r) {
  if (r.length < 3) return r;
  const f = r[0];
  const l = r[r.length - 1];
  if (f[0] !== l[0] || f[1] !== l[1]) return [...r, [...f]];
  return r;
}

function metersPerDegreeLat() { return 111320; }
function metersPerDegreeLng(latDeg) { return 111320 * Math.cos((latDeg * Math.PI) / 180); }

function createRectPolygon(centerLng, centerLat, halfWm = 20, halfHm = 12) {
  const dx = halfWm / metersPerDegreeLng(centerLat);
  const dy = halfHm / metersPerDegreeLat();
  return { type: "Polygon", coordinates: [[[centerLng - dx, centerLat - dy], [centerLng + dx, centerLat - dy], [centerLng + dx, centerLat + dy], [centerLng - dx, centerLat + dy], [centerLng - dx, centerLat - dy]]] };
}

const INFRA_FUNCTION_TYPES = new Set(["GROUND", "GREEN", "WALKWAY", "HIGHWAY"]);

function isInfraBlock(b) {
  const fn = String(b?.function || "").toUpperCase();
  return !!b?._infraSource || INFRA_FUNCTION_TYPES.has(fn);
}

function classifyUse(functionType) {
  const f = (functionType || "").toUpperCase();
  if (f === "RESIDENTIAL" || f === "APARTMENT") return "sale";
  if (f === "CENTER_OFFICE" || f === "SMALL_OFFICE" || f === "HOTEL" || f === "CENTER_COMMERCIAL" || f === "LEISURE_COMMERCIAL" || f === "OFFICE") return "rent";
  if (f === "PUBLIC") return "public_value";
  if (f === "GROUND" || f === "GREEN" || f === "WALKWAY" || f === "HIGHWAY") return "non_revenue";
  if (f === "COVER_MASS") return "cost_only";
  if (f.includes("RES")) return "sale";
  if (f.includes("OFFICE") || f.includes("COMM") || f.includes("RETAIL")) return "rent";
  if (f.includes("PUBLIC") || f.includes("GREEN") || f.includes("OPEN") || f.includes("ROAD") || f.includes("WALK")) return "non_revenue";
  return "mixed";
}

function getFunctionDefaults(functionType) {
  const f = (functionType || "").toUpperCase();
  if (f === "OFFICE") {
    return {
      saleable_ratio: 0.68,
      rentable_ratio: 0.82,
      cost_params: { hard_cost_per_sqm: 22800, soft_cost_ratio: 0.2, infra_cost_per_sqm: 2700, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 5550, occupancy: 0.9, opex_ratio: 0.24, cap_rate: 0.05 },
    };
  }
  if (f === "CENTER_OFFICE") {
    return {
      saleable_ratio: 0.7,
      rentable_ratio: 0.84,
      cost_params: { hard_cost_per_sqm: 23400, soft_cost_ratio: 0.2, infra_cost_per_sqm: 2700, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 7200, occupancy: 0.91, opex_ratio: 0.24, cap_rate: 0.05 },
    };
  }
  if (f === "SMALL_OFFICE") {
    return {
      saleable_ratio: 0.66,
      rentable_ratio: 0.8,
      cost_params: { hard_cost_per_sqm: 21000, soft_cost_ratio: 0.19, infra_cost_per_sqm: 2520, contingency_ratio: 0.09 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 5550, occupancy: 0.88, opex_ratio: 0.25, cap_rate: 0.052 },
    };
  }
  if (f === "APARTMENT") {
    return {
      saleable_ratio: 0.84,
      rentable_ratio: 0.15,
      cost_params: { hard_cost_per_sqm: 19500, soft_cost_ratio: 0.17, infra_cost_per_sqm: 2280, contingency_ratio: 0.08 },
      revenue_params: { sale_price_per_sqm: 126000, rent_price_per_sqm_year: 0, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
    };
  }
  if (f === "HOTEL") {
    return {
      saleable_ratio: 0.2,
      rentable_ratio: 0.82,
      cost_params: { hard_cost_per_sqm: 25800, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3300, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 6300, occupancy: 0.86, opex_ratio: 0.34, cap_rate: 0.056 },
    };
  }
  if (f === "RESIDENTIAL") {
    return {
      saleable_ratio: 0.82,
      rentable_ratio: 0.2,
      cost_params: { hard_cost_per_sqm: 20400, soft_cost_ratio: 0.18, infra_cost_per_sqm: 2400, contingency_ratio: 0.09 },
      revenue_params: { sale_price_per_sqm: 156000, rent_price_per_sqm_year: 0, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
    };
  }
  if (f === "CENTER_COMMERCIAL") {
    return {
      saleable_ratio: 0.7,
      rentable_ratio: 0.8,
      cost_params: { hard_cost_per_sqm: 26400, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3600, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 90000, rent_price_per_sqm_year: 7800, occupancy: 0.88, opex_ratio: 0.28, cap_rate: 0.052 },
    };
  }
  if (f === "LEISURE_COMMERCIAL") {
    return {
      saleable_ratio: 0.64,
      rentable_ratio: 0.76,
      cost_params: { hard_cost_per_sqm: 24600, soft_cost_ratio: 0.2, infra_cost_per_sqm: 3000, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 54000, rent_price_per_sqm_year: 4800, occupancy: 0.82, opex_ratio: 0.32, cap_rate: 0.058 },
    };
  }
  if (f === "GREEN") {
    return {
      saleable_ratio: 0,
      rentable_ratio: 0,
      cost_params: { hard_cost_per_sqm: 2700, soft_cost_ratio: 0.1, infra_cost_per_sqm: 2100, contingency_ratio: 0.06 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, occupancy: 0, opex_ratio: 0, cap_rate: 0.06 },
    };
  }
  if (f === "GROUND") {
    return {
      saleable_ratio: 0,
      rentable_ratio: 0,
      cost_params: { hard_cost_per_sqm: 3600, soft_cost_ratio: 0.1, infra_cost_per_sqm: 2700, contingency_ratio: 0.06 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, occupancy: 0, opex_ratio: 0, cap_rate: 0.06 },
    };
  }
  if (f === "WALKWAY") {
    return {
      saleable_ratio: 0,
      rentable_ratio: 0,
      cost_params: { hard_cost_per_sqm: 3600, soft_cost_ratio: 0.1, infra_cost_per_sqm: 3900, contingency_ratio: 0.06 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, occupancy: 0, opex_ratio: 0, cap_rate: 0.06 },
    };
  }
  if (f === "HIGHWAY") {
    return {
      saleable_ratio: 0,
      rentable_ratio: 0,
      cost_params: { hard_cost_per_sqm: 4500, soft_cost_ratio: 0.1, infra_cost_per_sqm: 5400, contingency_ratio: 0.08 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, occupancy: 0, opex_ratio: 0, cap_rate: 0.06 },
    };
  }
  if (f === "COVER_MASS") {
    return {
      saleable_ratio: 0.1,
      rentable_ratio: 0.2,
      cost_params: { hard_cost_per_sqm: 12600, soft_cost_ratio: 0.14, infra_cost_per_sqm: 2100, contingency_ratio: 0.08 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 1350, occupancy: 0.7, opex_ratio: 0.22, cap_rate: 0.06 },
    };
  }
  if (f === "PUBLIC") {
    return {
      saleable_ratio: 0.1,
      rentable_ratio: 0.2,
      cost_params: { hard_cost_per_sqm: 21000, soft_cost_ratio: 0.2, infra_cost_per_sqm: 2700, contingency_ratio: 0.1 },
      revenue_params: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 900, occupancy: 0.6, opex_ratio: 0.28, cap_rate: 0.06 },
    };
  }
  return {
    saleable_ratio: 0.78,
    rentable_ratio: 0.72,
    cost_params: { hard_cost_per_sqm: 15600, soft_cost_ratio: 0.18, infra_cost_per_sqm: 1800, contingency_ratio: 0.08 },
    revenue_params: { sale_price_per_sqm: 36000, rent_price_per_sqm_year: 3600, occupancy: 0.9, opex_ratio: 0.2, cap_rate: 0.05 },
  };
}

function n(v, d = 0) {
  const x = Number(v);
  return Number.isFinite(x) ? x : d;
}

export function createScenarioStore(buildingsGeojson) {
  let features = [];
  const byId = new Map();
  const undoStack = [];
  const redoStack = [];
  let idSeq = 1;

  const recalc = (item) => {
    const fn = item.feature.properties.functionType || "MIXED_USE";
    const fnDefaults = getFunctionDefaults(fn);
    item.metrics.footprint = polygonAreaSqm(item.feature.geometry);
    item.metrics.height = Number(item.feature.properties.Height || 12);
    item.metrics.base = Number(item.feature.properties.Base || 0);
    item.metrics.floors = Math.max(1, Math.round(item.metrics.height / 3.6));
    item.metrics.gfa = item.metrics.footprint * item.metrics.floors;
    item.metrics.saleable = item.metrics.gfa * Number(item.feature.properties.saleable_ratio ?? fnDefaults.saleable_ratio ?? 0.78);
    item.metrics.rentable = item.metrics.gfa * Number(item.feature.properties.rentable_ratio ?? fnDefaults.rentable_ratio ?? 0.72);
  };

  const snapshot = () => deepCopy(features.map((x) => ({ scenarioId: x.scenarioId, feature: x.feature, costParams: x.costParams, revenueParams: x.revenueParams })));
  const checkpoint = () => { undoStack.push(snapshot()); if (undoStack.length > 100) undoStack.shift(); redoStack.length = 0; };
  const rebuildById = () => { byId.clear(); features.forEach((item) => byId.set(item.scenarioId, item)); };
  const rebuildCollection = () => { buildingsGeojson.features = features.map((x) => x.feature); };

  const restore = (snap) => {
    features = snap.map((item) => {
      const r = { scenarioId: item.scenarioId, feature: item.feature, costParams: item.costParams, revenueParams: item.revenueParams, metrics: { footprint: 0, height: 0, base: 0, floors: 0, gfa: 0, saleable: 0, rentable: 0 } };
      recalc(r);
      return r;
    });
    rebuildById(); rebuildCollection();
  };

  const addFeature = (f, prefix = "bld") => {
    const scenarioId = String(f.properties?.id || f.properties?._scenarioId || `${prefix}_${idSeq++}`);
    f.properties = f.properties || {};
    f.properties._scenarioId = scenarioId;
    f.properties.Height = Number(f.properties.Height || 12);
    f.properties.Base = Number(f.properties.Base || 0);
    f.properties.functionType = f.properties.functionType || "CENTER_OFFICE";
    const defaults = getFunctionDefaults(f.properties.functionType);
    f.properties.saleable_ratio = Number(f.properties.saleable_ratio ?? defaults.saleable_ratio);
    f.properties.rentable_ratio = Number(f.properties.rentable_ratio ?? defaults.rentable_ratio);
    const item = {
      scenarioId,
      feature: f,
      metrics: { footprint: 0, height: 0, base: 0, floors: 0, gfa: 0, saleable: 0, rentable: 0 },
      costParams: f.properties.cost_params || defaults.cost_params,
      revenueParams: f.properties.revenue_params || defaults.revenue_params,
    };
    recalc(item); features.push(item); byId.set(item.scenarioId, item); return item.scenarioId;
  };

  const createItemFromBlock = (b, prefix = "infra") => {
    const f = {
      type: "Feature",
      properties: {
        id: b.id || `${prefix}_${idSeq++}`,
        Height: b.height ?? (String(b.function || "").toUpperCase() === "GREEN" ? 0.5 : 0.36),
        Base: b.base || 0,
        functionType: b.function || "WALKWAY",
        cost_params: b.cost_params,
        revenue_params: b.revenue_params,
        saleable_ratio: b.saleable_ratio,
        rentable_ratio: b.rentable_ratio,
      },
      geometry: b.geometry,
    };
    const scenarioId = String(f.properties.id);
    f.properties._scenarioId = scenarioId;
    const defaults = getFunctionDefaults(f.properties.functionType);
    f.properties.saleable_ratio = Number(f.properties.saleable_ratio ?? defaults.saleable_ratio);
    f.properties.rentable_ratio = Number(f.properties.rentable_ratio ?? defaults.rentable_ratio);
    const item = {
      scenarioId,
      feature: f,
      metrics: { footprint: 0, height: 0, base: 0, floors: 0, gfa: 0, saleable: 0, rentable: 0 },
      costParams: f.properties.cost_params || defaults.cost_params,
      revenueParams: f.properties.revenue_params || defaults.revenue_params,
    };
    recalc(item);
    return item;
  };

  let infraItems = [];

  buildingsGeojson.features.forEach((f, idx) => addFeature(f, `bld_${idx + 1}`));
  idSeq = features.length + 1;
  const initialSnapshot = snapshot();
  const resolveIds = (idsOrId) => Array.isArray(idsOrId) ? idsOrId : [idsOrId];

  return {
    getById(id) { return byId.get(id) || null; },
    getFeatureCollection() {
      rebuildCollection();
      if (!infraItems.length) return buildingsGeojson;
      return {
        type: "FeatureCollection",
        features: [...buildingsGeojson.features, ...infraItems.map((x) => x.feature)],
      };
    },
    loadScenarioJSON(scenario) {
      if (!scenario || !Array.isArray(scenario.blocks)) return false;
      checkpoint();
      features = []; byId.clear();
      infraItems = [];
      const blocks = scenario.blocks || [];
      blocks.filter((b) => !isInfraBlock(b)).forEach((b, idx) => {
        const f = { type: "Feature", properties: { id: b.id || `load_${idx + 1}`, Height: b.height || 24, Base: b.base || 0, functionType: b.function || "MIXED_USE", cost_params: b.cost_params, revenue_params: b.revenue_params, saleable_ratio: b.saleable_ratio, rentable_ratio: b.rentable_ratio }, geometry: b.geometry };
        addFeature(f, "load");
      });
      infraItems = blocks.filter((b) => isInfraBlock(b)).map((b, idx) => createItemFromBlock(b, `infra_${idx + 1}`));
      rebuildCollection();
      return true;
    },
    setInfraLayers(blocks = []) {
      const seen = new Set(infraItems.map((x) => x.scenarioId));
      (blocks || [])
        .filter((b) => b?.geometry?.type === "Polygon")
        .forEach((b, idx) => {
          const id = String(b.id || `infra_api_${idx + 1}`);
          if (seen.has(id)) return;
          seen.add(id);
          infraItems.push(createItemFromBlock({ ...b, id }, `infra_${idx + 1}`));
        });
    },
    resetToInitial() { restore(initialSnapshot); undoStack.length = 0; redoStack.length = 0; },
    importGeneratedBlocks(blocks) {
      checkpoint();
      blocks.forEach((b, idx) => {
        const f = { type: "Feature", properties: { id: b.id || `gen_${idx + 1}`, Height: b.height || 24, Base: b.base || 0, functionType: b.function || "CENTER_OFFICE", cost_params: b.cost_params, revenue_params: b.revenue_params, saleable_ratio: b.saleable_ratio, rentable_ratio: b.rentable_ratio }, geometry: b.geometry };
        addFeature(f, "gen");
      });
      rebuildCollection();
    },
    setHeight(ids, h) { checkpoint(); resolveIds(ids).forEach((id) => { const i = byId.get(id); if (i) { i.feature.properties.Height = Number(h); recalc(i); } }); },
    setBase(ids, b) { checkpoint(); resolveIds(ids).forEach((id) => { const i = byId.get(id); if (i) { i.feature.properties.Base = Number(b); recalc(i); } }); },
    setFunction(ids, f) {
      checkpoint();
      resolveIds(ids).forEach((id) => {
        const i = byId.get(id);
        if (!i) return;
        i.feature.properties.functionType = f;
        const d = getFunctionDefaults(f);
        i.feature.properties.saleable_ratio = d.saleable_ratio;
        i.feature.properties.rentable_ratio = d.rentable_ratio;
        i.costParams = deepCopy(d.cost_params);
        i.revenueParams = deepCopy(d.revenue_params);
        recalc(i);
      });
    },
    rotate(ids, angleDeg) { checkpoint(); resolveIds(ids).forEach((id) => { const i = byId.get(id); if (i) { rotateGeometry(i.feature.geometry, Number(angleDeg) || 0); recalc(i); } }); },
    translate(ids, deltaMetersLng, deltaMetersLat, latRef, withCheckpoint = false) { if (withCheckpoint) checkpoint(); const dx = deltaMetersLng / metersPerDegreeLng(latRef); const dy = deltaMetersLat / metersPerDegreeLat(); resolveIds(ids).forEach((id) => { const i = byId.get(id); if (i) { translateGeometry(i.feature.geometry, dx, dy); recalc(i); } }); },
    addBlockAt(lng, lat) { checkpoint(); const feature = { type: "Feature", properties: { Height: 24, Base: 0, functionType: "MIXED_USE" }, geometry: createRectPolygon(lng, lat) }; return addFeature(feature, "new_block"); },
    deleteBlocks(ids) { const arr = resolveIds(ids); if (!arr.length) return false; checkpoint(); const set = new Set(arr); features = features.filter((x) => !set.has(x.scenarioId)); rebuildById(); rebuildCollection(); return true; },
    updateVertex(id, vertexIndex, lng, lat, opts = { snapDeg: 0.00002, minEdgeDeg: 0.00002, withCheckpoint: false }) {
      const item = byId.get(id); if (!item || item.feature.geometry.type !== "Polygon") return false;
      if (opts.withCheckpoint) checkpoint();
      const ring = [...item.feature.geometry.coordinates[0]];
      const lastEditable = ring.length - 2;
      if (vertexIndex < 0 || vertexIndex > lastEditable) return false;
      const snap = opts.snapDeg || 0;
      const snappedLng = snap > 0 ? Math.round(lng / snap) * snap : lng;
      const snappedLat = snap > 0 ? Math.round(lat / snap) * snap : lat;
      ring[vertexIndex] = [snappedLng, snappedLat];
      const prev = ring[(vertexIndex - 1 + ring.length - 1) % (ring.length - 1)];
      const next = ring[(vertexIndex + 1) % (ring.length - 1)];
      const d1 = Math.hypot(prev[0] - snappedLng, prev[1] - snappedLat);
      const d2 = Math.hypot(next[0] - snappedLng, next[1] - snappedLat);
      if (d1 < (opts.minEdgeDeg || 0) || d2 < (opts.minEdgeDeg || 0)) return false;
      item.feature.geometry.coordinates[0] = upsertClosedRing(ring.slice(0, -1));
      recalc(item);
      return true;
    },
    getVertexMarkers(id) { const item = byId.get(id); if (!item || item.feature.geometry.type !== "Polygon") return { type: "FeatureCollection", features: [] }; const ring = item.feature.geometry.coordinates[0] || []; return { type: "FeatureCollection", features: ring.slice(0, -1).map((coord, idx) => ({ type: "Feature", properties: { idx }, geometry: { type: "Point", coordinates: coord } })) }; },
    undo() { if (!undoStack.length) return false; redoStack.push(snapshot()); restore(undoStack.pop()); return true; },
    redo() { if (!redoStack.length) return false; undoStack.push(snapshot()); restore(redoStack.pop()); return true; },
    getStats(params = {}) {
      const totals = {
        blocks: features.length,
        footprint: 0,
        gfa: 0,
        saleable_area: 0,
        rentable_area: 0,
        gdv: 0,
        noi: 0,
        cap_value: 0,
        market_value: 0,
        total_value: 0,
        public_value: 0,
        tdc: 0,
        profit: 0,
        profit_on_cost: 0,
        profit_on_gdv: 0,
        direct_cost: 0,
        land_cost: 0,
        finance_cost: 0,
        tax_cost: 0,
        sale_marketing_cost: 0,
        lease_marketing_cost: 0,
        residual_land_value: 0,
        peak_funding: 0,
        payback_year: null,
        by_function: {},
        by_function_financial: {},
        cost_stack: { hard: 0, infra: 0, soft: 0, contingency: 0, land: 0, tax: 0, sale_marketing: 0, lease_marketing: 0, finance: 0 },
        yearly: [],
        data_warning: null,
        eco: {
          ground_area_sqm: 0,
          green_area_sqm: 0,
          carbon_sink_tpy: 0,
          runoff_reduction_m3y: 0,
          cooling_score: 0,
          eco_value: 0,
        },
        composite_score: 0,
        composite_components: { econ_score: 0, eco_score: 0 },
      };

      const project = {
        years: Math.max(3, n(params.years, 5)),
        buildYears: Math.max(1, Math.min(Math.max(3, n(params.years, 5)) - 1, n(params.buildYears, 2))),
        landCostPerGfa: Math.max(0, n(params.landCostPerGfa, 18000)),
        financeRate: Math.max(0, n(params.financeRate, 0.055)),
        debtRatio: Math.max(0, Math.min(1, n(params.debtRatio, 0.55))),
        taxRatio: Math.max(0, n(params.taxRatio, 0.06)),
        saleMarketingRatio: Math.max(0, n(params.saleMarketingRatio, 0.03)),
        leaseMarketingRatio: Math.max(0, n(params.leaseMarketingRatio, 0.02)),
        targetProfitRatio: Math.max(0, n(params.targetProfitRatio, 0.15)),
        publicValuePerSqm: Math.max(0, n(params.publicValuePerSqm, 2200)),
      };

      let eligibleLandGfa = 0;
      const revenueRows = [];
      const financeRows = [];
      const statsItems = [...features, ...infraItems];

      statsItems.forEach((item) => {
        const fn = item.feature.properties.functionType || "MIXED_USE";
        const kind = classifyUse(fn);
        const defaults = getFunctionDefaults(fn);
        const defaultCost = defaults.cost_params || {};
        const defaultRevenue = defaults.revenue_params || {};

        const fnOverride = params.functionOverrides?.[fn] || {};
        const hard = Math.max(0, n(fnOverride.hard_cost_per_sqm ?? defaultCost.hard_cost_per_sqm, 5200));
        const soft = Math.max(0, n(fnOverride.soft_cost_ratio ?? defaultCost.soft_cost_ratio, 0.18));
        const infra = Math.max(0, n(fnOverride.infra_cost_per_sqm ?? defaultCost.infra_cost_per_sqm, 600));
        const cont = Math.max(0, n(fnOverride.contingency_ratio ?? defaultCost.contingency_ratio, 0.08));

        const saleP = Math.max(0, n(fnOverride.sale_price_per_sqm ?? defaultRevenue.sale_price_per_sqm, 12000));
        const rentY = Math.max(0, n(fnOverride.rent_price_per_sqm_year ?? defaultRevenue.rent_price_per_sqm_year, 1200));
        const occ = Math.max(0, Math.min(1, n(fnOverride.occupancy ?? defaultRevenue.occupancy, 0.9)));
        const opex = Math.max(0, Math.min(1, n(fnOverride.opex_ratio ?? defaultRevenue.opex_ratio, 0.2)));
        const cap = Math.max(0.0001, n(fnOverride.cap_rate ?? defaultRevenue.cap_rate, 0.05));

        totals.gfa += item.metrics.gfa;
        totals.footprint += item.metrics.footprint;
        totals.saleable_area += item.metrics.saleable;
        totals.rentable_area += item.metrics.rentable;
        totals.by_function[fn] = (totals.by_function[fn] || 0) + item.metrics.gfa;
        if (!totals.by_function_financial[fn]) {
          totals.by_function_financial[fn] = { gfa: 0, saleable_area: 0, rentable_area: 0, revenue: 0, cost: 0, profit: 0 };
        }
        totals.by_function_financial[fn].gfa += item.metrics.gfa;
        totals.by_function_financial[fn].saleable_area += item.metrics.saleable;
        totals.by_function_financial[fn].rentable_area += item.metrics.rentable;

        const hardCost = item.metrics.gfa * hard;
        const infraCost = item.metrics.gfa * infra;
        const softCost = hardCost * soft;
        const contingency = hardCost * cont;
        const directCost = hardCost + infraCost + softCost + contingency;
        totals.direct_cost += directCost;
        totals.cost_stack.hard += hardCost;
        totals.cost_stack.infra += infraCost;
        totals.cost_stack.soft += softCost;
        totals.cost_stack.contingency += contingency;
        totals.by_function_financial[fn].cost += directCost;
        totals.by_function_financial[fn].direct_cost = (totals.by_function_financial[fn].direct_cost || 0) + directCost;
        if (fn === "GREEN") {
          totals.eco.green_area_sqm += item.metrics.footprint;
        }
        if (fn === "GROUND") {
          totals.eco.ground_area_sqm += item.metrics.footprint;
        }
        if (kind !== "non_revenue") {
          eligibleLandGfa += item.metrics.gfa;
        }
        financeRows.push({ fn, base: directCost });

        if (kind === "sale") {
          const rev = item.metrics.saleable * saleP;
          totals.gdv += rev;
          totals.by_function_financial[fn].revenue += rev;
          totals.by_function_financial[fn].sale_revenue = (totals.by_function_financial[fn].sale_revenue || 0) + rev;
          totals.by_function_financial[fn].market_revenue = (totals.by_function_financial[fn].market_revenue || 0) + rev;
          revenueRows.push({ fn, market: rev, sale: rev, lease: 0, publicValue: 0 });
        } else if (kind === "rent") {
          const egi = item.metrics.rentable * rentY * occ;
          const noi = egi * (1 - opex);
          totals.noi += noi;
          const rev = noi / cap;
          totals.cap_value += rev;
          totals.by_function_financial[fn].revenue += rev;
          totals.by_function_financial[fn].lease_value = (totals.by_function_financial[fn].lease_value || 0) + rev;
          totals.by_function_financial[fn].market_revenue = (totals.by_function_financial[fn].market_revenue || 0) + rev;
          revenueRows.push({ fn, market: rev, sale: 0, lease: rev, publicValue: 0 });
        } else if (kind === "public_value") {
          const serviceValue = item.metrics.gfa * project.publicValuePerSqm;
          totals.public_value += serviceValue;
          totals.by_function_financial[fn].revenue += serviceValue;
          totals.by_function_financial[fn].public_value = (totals.by_function_financial[fn].public_value || 0) + serviceValue;
          revenueRows.push({ fn, market: 0, sale: 0, lease: 0, publicValue: serviceValue });
        } else if (kind === "non_revenue" || kind === "cost_only") {
          revenueRows.push({ fn, market: 0, sale: 0, lease: 0, publicValue: 0 });
        } else {
          const saleRev = item.metrics.saleable * saleP * 0.5;
          const egi = item.metrics.rentable * rentY * occ * 0.5;
          const noi = egi * (1 - opex);
          totals.gdv += saleRev;
          totals.noi += noi;
          const capRev = noi / cap;
          totals.cap_value += capRev;
          totals.by_function_financial[fn].revenue += saleRev + capRev;
          totals.by_function_financial[fn].sale_revenue = (totals.by_function_financial[fn].sale_revenue || 0) + saleRev;
          totals.by_function_financial[fn].lease_value = (totals.by_function_financial[fn].lease_value || 0) + capRev;
          totals.by_function_financial[fn].market_revenue = (totals.by_function_financial[fn].market_revenue || 0) + saleRev + capRev;
          revenueRows.push({ fn, market: saleRev + capRev, sale: saleRev, lease: capRev, publicValue: 0 });
        }
      });

      totals.land_cost = eligibleLandGfa * project.landCostPerGfa;
      totals.cost_stack.land = totals.land_cost;

      const financeBase = totals.direct_cost + totals.land_cost;
      totals.finance_cost = financeBase * project.debtRatio * project.financeRate * (project.buildYears / 2);
      totals.cost_stack.finance = totals.finance_cost;

      totals.market_value = totals.gdv + totals.cap_value;
      totals.tax_cost = totals.market_value * project.taxRatio;
      totals.cost_stack.tax = totals.tax_cost;

      totals.sale_marketing_cost = totals.gdv * project.saleMarketingRatio;
      totals.lease_marketing_cost = totals.cap_value * project.leaseMarketingRatio;
      totals.cost_stack.sale_marketing = totals.sale_marketing_cost;
      totals.cost_stack.lease_marketing = totals.lease_marketing_cost;

      const landShareDen = Math.max(1, eligibleLandGfa);
      const marketShareDen = Math.max(1, totals.market_value);
      const saleShareDen = Math.max(1, totals.gdv);
      const leaseShareDen = Math.max(1, totals.cap_value);
      const financeShareDen = Math.max(1, financeBase);

      Object.keys(totals.by_function_financial).forEach((fn) => {
        const fin = totals.by_function_financial[fn];
        const landShare = fin.gfa / landShareDen;
        const marketShare = Number(fin.market_revenue || 0) / marketShareDen;
        const saleShare = Number(fin.sale_revenue || 0) / saleShareDen;
        const leaseShare = Number(fin.lease_value || 0) / leaseShareDen;
        const financeBaseShare = ((Number(fin.direct_cost || 0) + (totals.land_cost * landShare)) / financeShareDen);
        const landAlloc = totals.land_cost * landShare;
        const taxAlloc = totals.tax_cost * marketShare;
        const saleMarketingAlloc = totals.sale_marketing_cost * saleShare;
        const leaseMarketingAlloc = totals.lease_marketing_cost * leaseShare;
        const financeAlloc = totals.finance_cost * financeBaseShare;
        fin.land_cost = landAlloc;
        fin.tax_cost = taxAlloc;
        fin.sale_marketing_cost = saleMarketingAlloc;
        fin.lease_marketing_cost = leaseMarketingAlloc;
        fin.finance_cost = financeAlloc;
        fin.cost += landAlloc + taxAlloc + saleMarketingAlloc + leaseMarketingAlloc + financeAlloc;
      });

      totals.total_value = totals.market_value + totals.public_value;
      totals.tdc = totals.direct_cost + totals.land_cost + totals.finance_cost + totals.tax_cost + totals.sale_marketing_cost + totals.lease_marketing_cost;
      totals.profit = totals.total_value - totals.tdc;
      totals.profit_on_cost = totals.tdc > 0 ? totals.profit / totals.tdc : 0;
      totals.profit_on_gdv = totals.total_value > 0 ? totals.profit / totals.total_value : 0;
      totals.residual_land_value = totals.total_value - (totals.tdc - totals.land_cost) - (totals.total_value * project.targetProfitRatio);
      Object.keys(totals.by_function_financial).forEach((k) => {
        totals.by_function_financial[k].profit = totals.by_function_financial[k].revenue - totals.by_function_financial[k].cost;
      });

      const years = project.years;
      const buildYears = project.buildYears;
      const leaseYears = Math.max(1, years - buildYears);

      // GDV (sale) recognised in first post-construction year; NOI (rent) recurs annually.
      const annualNOI = totals.noi;
      let cum = 0;
      let minCum = 0;
      for (let y = 1; y <= years; y += 1) {
        const annualCost = y <= buildYears ? (totals.direct_cost + totals.land_cost + totals.finance_cost) / buildYears : 0;
        const taxAndMkt = totals.tax_cost + totals.sale_marketing_cost + totals.lease_marketing_cost;
        const annualRevenue = y > buildYears
          ? annualNOI + (y === buildYears + 1 ? totals.gdv + totals.public_value - taxAndMkt : 0)
          : 0;
        const net = annualRevenue - annualCost;
        cum += net;
        minCum = Math.min(minCum, cum);
        if (totals.payback_year === null && cum >= 0) totals.payback_year = y;
        totals.yearly.push({ year: y, annual_cost: annualCost, annual_revenue: annualRevenue, cashflow: net, cumulative: cum });
      }
      totals.peak_funding = Math.abs(minCum);

      if (totals.blocks > 0 && totals.gfa <= 0) {
        totals.data_warning = "Geometry not connected to economics model / 体块面积未传入经济模型";
      }

      // Lightweight ecological benefit proxies for GREEN areas.
      const greenArea = totals.eco.green_area_sqm;
      const ecoCarbonFactor = n(params.ecoCarbonFactor, 0.0009);
      const ecoRunoffFactor = n(params.ecoRunoffFactor, 0.65);
      const ecoCarbonPrice = n(params.ecoCarbonPrice, 520);
      const ecoRunoffPrice = n(params.ecoRunoffPrice, 3.2);
      totals.eco.carbon_sink_tpy = greenArea * ecoCarbonFactor; // tCO2e / year
      totals.eco.runoff_reduction_m3y = greenArea * ecoRunoffFactor; // m3 / year
      totals.eco.cooling_score = Math.min(100, (greenArea / Math.max(1, totals.footprint)) * 140);
      totals.eco.eco_value = totals.eco.carbon_sink_tpy * ecoCarbonPrice + totals.eco.runoff_reduction_m3y * ecoRunoffPrice; // CNY/year proxy

      const econScoreRaw = 50 + (totals.profit_on_cost * 120);
      const ecoScoreRaw = (totals.eco.cooling_score * 0.6) + (Math.min(100, (totals.eco.eco_value / 1e6) * 25) * 0.4);
      const econScore = Math.max(0, Math.min(100, econScoreRaw));
      const ecoScore = Math.max(0, Math.min(100, ecoScoreRaw));
      totals.composite_score = (econScore * 0.7) + (ecoScore * 0.3);
      totals.composite_components = { econ_score: econScore, eco_score: ecoScore };

      const scalarKeys = [
        "footprint", "gfa", "saleable_area", "rentable_area", "gdv", "noi", "cap_value", "market_value", "total_value",
        "public_value", "tdc", "profit", "profit_on_cost", "profit_on_gdv", "direct_cost", "land_cost", "finance_cost",
        "tax_cost", "sale_marketing_cost", "lease_marketing_cost", "residual_land_value", "peak_funding",
      ];
      scalarKeys.forEach((k) => { totals[k] = n(totals[k], 0); });
      totals.composite_score = n(totals.composite_score, 0);
      totals.composite_components.econ_score = n(totals.composite_components.econ_score, 0);
      totals.composite_components.eco_score = n(totals.composite_components.eco_score, 0);
      totals.eco.green_area_sqm = n(totals.eco.green_area_sqm, 0);
      totals.eco.ground_area_sqm = n(totals.eco.ground_area_sqm, 0);
      totals.eco.carbon_sink_tpy = n(totals.eco.carbon_sink_tpy, 0);
      totals.eco.runoff_reduction_m3y = n(totals.eco.runoff_reduction_m3y, 0);
      totals.eco.cooling_score = n(totals.eco.cooling_score, 0);
      totals.eco.eco_value = n(totals.eco.eco_value, 0);
      totals.yearly = totals.yearly.map((r) => ({
        year: n(r.year, 0),
        annual_cost: n(r.annual_cost, 0),
        annual_revenue: n(r.annual_revenue, 0),
        cashflow: n(r.cashflow, 0),
        cumulative: n(r.cumulative, 0),
      }));

      return totals;
    },
    toScenarioJSON(name = "scenario_mvp", econParams = null) {
      return {
        schema_version: "0.2.0",
        scenario_name: name,
        generated_at: new Date().toISOString(),
        economics_params: econParams || undefined,
        blocks: features.map((item) => ({
          ...(function buildEconomicsForExport() {
            const defaults = getFunctionDefaults(item.feature.properties.functionType || "MIXED_USE");
            const override = econParams?.functionOverrides?.[item.feature.properties.functionType || ""] || {};
            return {
              cost_params: {
                hard_cost_per_sqm: n(override.hard_cost_per_sqm ?? defaults.cost_params?.hard_cost_per_sqm, 0),
                soft_cost_ratio: n(override.soft_cost_ratio ?? defaults.cost_params?.soft_cost_ratio, 0),
                infra_cost_per_sqm: n(override.infra_cost_per_sqm ?? defaults.cost_params?.infra_cost_per_sqm, 0),
                contingency_ratio: n(override.contingency_ratio ?? defaults.cost_params?.contingency_ratio, 0),
              },
              revenue_params: {
                sale_price_per_sqm: n(override.sale_price_per_sqm ?? defaults.revenue_params?.sale_price_per_sqm, 0),
                rent_price_per_sqm_year: n(override.rent_price_per_sqm_year ?? defaults.revenue_params?.rent_price_per_sqm_year, 0),
                occupancy: n(override.occupancy ?? defaults.revenue_params?.occupancy, 0),
                opex_ratio: n(override.opex_ratio ?? defaults.revenue_params?.opex_ratio, 0),
                cap_rate: n(override.cap_rate ?? defaults.revenue_params?.cap_rate, 0),
              },
            };
          }()),
          id: item.scenarioId,
          geometry: item.feature.geometry,
          function: item.feature.properties.functionType || "MIXED_USE",
          gfa: item.metrics.gfa,
          footprint: item.metrics.footprint,
          saleable_area: item.metrics.saleable,
          rentable_area: item.metrics.rentable,
          height: item.metrics.height,
          base: item.metrics.base,
          floors: item.metrics.floors,
          saleable_ratio: item.feature.properties.saleable_ratio,
          rentable_ratio: item.feature.properties.rentable_ratio,
        })),
      };
    },
  };
}
