import { DATASETS } from "./config.js";
import { createMap } from "./map-init.js";
import { addSourcesAndLayers, fitToSiteBoundary, refreshBuildingStackSource, setLayerVisible, setSelectedParcel } from "./layer-manager.js";
import { loadDatasets } from "./source-loader.js";
import { bindInfoPopups } from "./popup-controller.js";
import { bindLayerToggles, initEditorUI, initLegends, renderSummary, setStatus } from "./ui-controller.js";
import { createScenarioStore } from "./scenario-model.js";
import { createEditorController } from "./editor-controller.js";
import { requestClusterGeneration } from "./cluster/cluster-client.js";
import { requestAudienceCompletion, requestFloorPlanWithAgent } from "./ai-agent-client.js";
import { createZoneInsightOverlay } from "./zone-insight-overlay.js";

const SCENARIO_CACHE_KEY = "site_design_platform_scenario";
const ECON_PARAMS_CACHE_KEY = "site_design_platform_econ_params";
const RHINO_REVISION_KEY = "site_design_platform_rhino_revision";

function toPlainFeature(feature) {
  if (!feature || !feature.geometry) return null;
  return {
    type: "Feature",
    properties: JSON.parse(JSON.stringify(feature.properties || {})),
    geometry: JSON.parse(JSON.stringify(feature.geometry)),
  };
}

function downloadJSON(fileName, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

function numOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function collectLngLatBoundsFromFeatureCollection(fc) {
  const bounds = new maplibregl.LngLatBounds();
  let count = 0;
  (fc?.features || []).forEach((f) => {
    const g = f?.geometry;
    if (!g) return;
    const addCoord = (c) => {
      if (!Array.isArray(c) || c.length < 2) return;
      const lng = Number(c[0]);
      const lat = Number(c[1]);
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) return;
      bounds.extend([lng, lat]);
      count += 1;
    };
    if (g.type === "Polygon") {
      (g.coordinates || []).forEach((ring) => (ring || []).forEach(addCoord));
    } else if (g.type === "MultiPolygon") {
      (g.coordinates || []).forEach((poly) => (poly || []).forEach((ring) => (ring || []).forEach(addCoord)));
    }
  });
  return { bounds, count };
}

function pointInRing(lng, lat, ring) {
  let inside = false;
  let j = ring.length - 1;
  for (let i = 0; i < ring.length; i += 1) {
    const xi = ring[i][0]; const yi = ring[i][1];
    const xj = ring[j][0]; const yj = ring[j][1];
    const hit = ((yi > lat) !== (yj > lat)) && (lng < ((xj - xi) * (lat - yi)) / ((yj - yi) || 1e-12) + xi);
    if (hit) inside = !inside;
    j = i;
  }
  return inside;
}

function centroidOfPolygonFeature(f) {
  const ring = f?.geometry?.type === "Polygon" ? (f.geometry.coordinates[0] || []) : [];
  if (!ring.length) return null;
  let x = 0; let y = 0; let n = 0;
  ring.slice(0, -1).forEach((p) => { x += p[0]; y += p[1]; n += 1; });
  return n ? [x / n, y / n] : ring[0];
}

function polygonAreaSqmFromFeature(f) {
  const ring = f?.geometry?.type === "Polygon" ? (f.geometry.coordinates[0] || []) : [];
  if (ring.length < 4) return 0;
  const lat0 = ring.reduce((a, c) => a + c[1], 0) / ring.length;
  let s = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const x1 = ring[i][0] * 111320 * Math.cos((lat0 * Math.PI) / 180);
    const y1 = ring[i][1] * 111320;
    const x2 = ring[i + 1][0] * 111320 * Math.cos((lat0 * Math.PI) / 180);
    const y2 = ring[i + 1][1] * 111320;
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

function buildStackExtrusions(plan, store) {
  const floorH = 3.6;
  const sliceFloors = 3;
  const features = [];
  (plan?.outputs || []).forEach((o) => {
    const item = store.getById(o.id);
    if (!item) return;
    const totalFloors = Math.max(1, Math.round(Number(item.metrics?.floors || 1)));
    const segs = (o.segments || []).length ? o.segments : [{ segment: "full", primaryCode: o.dominantCode || "PUBLIC", primary: o.dominant || "公共服务", secondary: null, score: 0 }];
    const slices = Math.max(1, Math.ceil(totalFloors / sliceFloors));
    for (let i = 0; i < slices; i += 1) {
      const fStart = i * sliceFloors + 1;
      const fEnd = Math.min(totalFloors, (i + 1) * sliceFloors);
      const n = fEnd - fStart + 1;
      const t = slices === 1 ? 0 : i / (slices - 1);
      const segIdx = Math.max(0, Math.min(segs.length - 1, Math.floor(t * segs.length)));
      const seg = segs[segIdx];
      const base = (fStart - 1) * floorH;
      const height = n * floorH;
      features.push({
        type: "Feature",
        properties: {
          _scenarioId: o.id,
          segment: `${fStart}-${fEnd}F`,
          floorStart: fStart,
          floorEnd: fEnd,
          primary: seg.primary || "",
          secondary: seg.secondary || "",
          functionCode: seg.primaryCode || o.dominantCode || "PUBLIC",
          Base: base,
          Height: height,
          score: Number(seg.score || 0),
          reason: seg.reason || "",
        },
        geometry: item.feature.geometry,
      });
    }
  });
  return { type: "FeatureCollection", features };
}

function inferPublicSubtype(prompt, zone = "UNKNOWN") {
  const t = String(prompt || "");
  const rules = [
    { key: "剧院演艺", re: /(剧场|剧院|演艺|戏剧|秀场|表演)/, zones: ["LEISURE", "CREATIVE"] },
    { key: "会展会议", re: /(会展|会议|论坛|发布会|博览)/, zones: ["CBD", "CREATIVE"] },
    { key: "博物馆", re: /(博物馆|馆藏|策展|历史展)/, zones: ["CREATIVE", "LEISURE"] },
    { key: "科普馆", re: /(科普|科技馆|科学|实验教育)/, zones: ["CREATIVE", "RESIDENTIAL"] },
    { key: "城市展厅", re: /(城市客厅|展示中心|城市展厅|地标叙事)/, zones: ["CBD", "LEISURE"] },
    { key: "社区服务中心", re: /(社区|邻里|家庭|便民|公共服务)/, zones: ["RESIDENTIAL"] },
    { key: "青少年活动中心", re: /(青少年|学生|研学|教育活动)/, zones: ["RESIDENTIAL", "CREATIVE"] },
  ];
  let best = { key: "综合公共服务", score: 1 };
  rules.forEach((r) => {
    let s = 0;
    if (r.re.test(t)) s += 3;
    if (r.zones.includes(zone)) s += 1;
    if (s > best.score) best = { key: r.key, score: s };
  });
  return best.key;
}

function renderParcelMetrics(parcelsFc, blocksFc) {
  const summaryEl = document.getElementById("eco-parcel-summary");
  const tableEl = document.getElementById("eco-parcel-table");
  if (!summaryEl || !tableEl) return;
  const parcels = (parcelsFc?.features || []).filter((f) => f?.geometry?.type === "Polygon");
  if (!parcels.length) {
    summaryEl.textContent = "未读取到地块数据。";
    tableEl.innerHTML = "";
    return;
  }
  const blocks = (blocksFc?.features || []).filter((f) => f?.geometry?.type === "Polygon");
  const rows = parcels.map((p) => {
    const ring = p.geometry.coordinates[0] || [];
    const land = polygonAreaSqmFromFeature(p);
    const farLimit = Number(p.properties?.far_limit ?? 4.0);
    const hLimit = Number(p.properties?.height_limit ?? 120);
    let gfa = 0;
    let hMax = 0;
    let greenArea = 0;
    blocks.forEach((b) => {
      const c = centroidOfPolygonFeature(b);
      if (!c) return;
      if (!pointInRing(c[0], c[1], ring)) return;
      const fp = polygonAreaSqmFromFeature(b);
      const h = Number(b.properties?.Height ?? b.properties?.height ?? 24);
      const floors = Math.max(1, Math.round(h / 3.6));
      gfa += fp * floors;
      hMax = Math.max(hMax, h);
      const fn = String(b.properties?.functionType || b.properties?.function || "").toUpperCase();
      if (fn === "GREEN") greenArea += fp;
    });
    const farActual = land > 0 ? gfa / land : 0;
    const farUse = farLimit > 0 ? farActual / farLimit : 0;
    const hUse = hLimit > 0 ? hMax / hLimit : 0;
    const status = farUse > 1 || hUse > 1 ? "超标" : (farUse > 0.9 || hUse > 0.9 ? "临界" : "合规");
    const greenRatio = land > 0 ? greenArea / land : 0;
    const ecoPotential = Math.max(0, Math.min(100, greenRatio * 250));
    return {
      id: String(p.properties?.parcel_id || p.properties?.id || "-"),
      zone: String(p.properties?.zone_id || p.properties?.layer || "-"),
      land, gfa, farLimit, farActual, hLimit, hMax, status, farUse, hUse, greenArea, greenRatio, ecoPotential,
    };
  });
  const overCount = rows.filter((r) => r.status === "超标").length;
  const totalLand = rows.reduce((a, r) => a + r.land, 0);
  const totalGfa = rows.reduce((a, r) => a + r.gfa, 0);
  summaryEl.innerHTML = `地块数 ${rows.length} | 用地总面积 ${Math.round(totalLand).toLocaleString()}㎡ | 建筑总面积 ${Math.round(totalGfa).toLocaleString()}㎡ | 超标地块 ${overCount}`;
  tableEl.innerHTML = rows.map((r) => {
    const color = r.status === "超标" ? "#ff6b6b" : (r.status === "临界" ? "#ffd166" : "#31c48d");
    const farUseSafe = Number.isFinite(r.farUse) ? r.farUse : 0;
    const hUseSafe = Number.isFinite(r.hUse) ? r.hUse : 0;
    const farPct = Math.max(0, Math.min(140, Math.round(farUseSafe * 100)));
    const hPct = Math.max(0, Math.min(140, Math.round(hUseSafe * 100)));
    const farBar = `<div style="height:6px;background:#1b2738;border-radius:4px;overflow:hidden;"><div style="width:${Math.min(farPct, 100)}%;height:6px;background:${farUseSafe > 1 ? "#ff6b6b" : "#36c5f0"}"></div></div>`;
    const hBar = `<div style="height:6px;background:#1b2738;border-radius:4px;overflow:hidden;"><div style="width:${Math.min(hPct, 100)}%;height:6px;background:${hUseSafe > 1 ? "#ff6b6b" : "#7ad97a"}"></div></div>`;
    return `<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,.08)">
      <div><b>${r.zone}</b> / ${r.id} <span style="color:${color};font-weight:600">[${r.status}]</span></div>
      <div>用地 ${Math.round(r.land).toLocaleString()}㎡ | 建面 ${Math.round(r.gfa).toLocaleString()}㎡</div>
      <div>FAR ${r.farActual.toFixed(2)} / ${r.farLimit.toFixed(2)} | 限高 ${r.hMax.toFixed(1)}m / ${r.hLimit.toFixed(1)}m</div>
      <div>绿地 ${Math.round(r.greenArea).toLocaleString()}㎡ (${(r.greenRatio * 100).toFixed(1)}%) | 生态潜力 ${r.ecoPotential.toFixed(0)}/100</div>
      <div style="margin-top:3px;font-size:11px;color:#a9bddc;">FAR使用率 ${farPct}%</div>
      ${farBar}
      <div style="margin-top:3px;font-size:11px;color:#a9bddc;">限高使用率 ${hPct}%</div>
      ${hBar}
    </div>`;
  }).join("");
}

async function fetchRhinoScenario() {
  const r = await fetch("/api/site-design/rhino/latest", { cache: "no-store" });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`rhino latest failed: ${r.status} ${t}`);
  }
  const scenario = await r.json();
  return {
    scenario,
    updatedAt: Number(r.headers.get("X-Rhino-Updated-At") || 0),
    blocksCount: Number(r.headers.get("X-Rhino-Blocks-Count") || 0),
  };
}

async function fetchRhinoParcels() {
  const r = await fetch("/api/site-design/rhino/parcels");
  if (!r.ok) return { type: "FeatureCollection", features: [] };
  return r.json();
}

async function fetchRhinoOriginalBuildings() {
  const r = await fetch("/api/site-design/rhino/original-buildings");
  if (!r.ok) return { type: "FeatureCollection", features: [] };
  return r.json();
}

async function fetchRhinoWalking() {
  const r = await fetch("/api/site-design/rhino/walking");
  if (!r.ok) return { type: "FeatureCollection", features: [] };
  return r.json();
}

async function fetchRhinoGround() {
  const r = await fetch("/api/site-design/rhino/ground");
  if (!r.ok) return { type: "FeatureCollection", features: [] };
  return r.json();
}

async function saveEconomicsSnapshot(payload) {
  const r = await fetch("/api/site-design/economics/snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`save snapshot failed: ${r.status} ${t}`);
  }
  return r.json();
}

async function main() {
  initLegends();
  setStatus("Loading GeoJSON layers...");
  const datasets = await loadDatasets(DATASETS);
  const map = createMap();
  const store = createScenarioStore(datasets.buildings);
  let selectedParcel = null;
  let cachedEconParams = null;
  let currentEconParams = null;
  let rhinoBootstrap = null;
  let lastRhinoUpdatedAt = 0;

  try {
    const probe = await fetchRhinoScenario();
    if (probe.scenario && Array.isArray(probe.scenario.blocks) && probe.scenario.blocks.length > 0) {
      rhinoBootstrap = probe;
      lastRhinoUpdatedAt = probe.updatedAt || 0;
    }
  } catch {
    // Rhino unavailable — fall back to local cache below.
  }

  if (!rhinoBootstrap) {
    const cached = localStorage.getItem(SCENARIO_CACHE_KEY);
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (store.loadScenarioJSON(parsed)) setStatus("Recovered last scenario from local cache");
      } catch {
        // ignore cache parse errors
      }
    }
  }
  const cachedEcon = localStorage.getItem(ECON_PARAMS_CACHE_KEY);
  if (cachedEcon) {
    try { cachedEconParams = JSON.parse(cachedEcon); } catch { cachedEconParams = null; }
  }

  map.on("load", () => {
    addSourcesAndLayers(map, datasets);
    setLayerVisible(map, "buildings-stack-extrusion", false);
    fitToSiteBoundary(map, datasets.siteBoundary);
    bindLayerToggles(map);
    bindInfoPopups(map);
    renderSummary(datasets.summary);

    const persist = () => {
      const payload = store.toScenarioJSON("scenario_mvp", currentEconParams);
      if (lastRhinoUpdatedAt > 0) {
        payload.scenario_source = "rhino";
        payload.rhino_updated_at = lastRhinoUpdatedAt;
      }
      localStorage.setItem(SCENARIO_CACHE_KEY, JSON.stringify(payload));
      if (lastRhinoUpdatedAt > 0) {
        localStorage.setItem(RHINO_REVISION_KEY, String(lastRhinoUpdatedAt));
      }
    };
    const zoneOverlay = createZoneInsightOverlay(map);
    const refreshParcelMetricsFromMap = () => {
      const src = map.getSource("zone-parcels");
      const fc = src && src._data ? src._data : datasets.zoneParcels;
      renderParcelMetrics(fc, store.getFeatureCollection());
    };

    map.on("click", "zone-parcels-fill", (e) => {
      selectedParcel = toPlainFeature(e.features?.[0]);
      setSelectedParcel(map, selectedParcel);
      const zid = selectedParcel?.properties?.zone_id || "ZONE";
      const pid = selectedParcel?.properties?.parcel_id || selectedParcel?.properties?.id || "-";
      document.getElementById("selected-parcel-id").textContent = `${zid}:${pid}`;
      setStatus(`Selected parcel ${zid}:${pid}`);
    });

    map.on("click", (e) => {
      const hits = map.queryRenderedFeatures(e.point, { layers: ["zone-parcels-fill"] });
      if (!hits.length) {
        selectedParcel = null;
        setSelectedParcel(map, null);
        document.getElementById("selected-parcel-id").textContent = "-";
      }
    });
    map.on("click", "buildings-stack-extrusion", (e) => {
      const p = e.features?.[0]?.properties || {};
      const msg = `体块 ${p._scenarioId || "-"} | 楼层 ${p.segment || "-"}\n主功能: ${p.primary || p.functionCode || "-"}\n次功能: ${p.secondary || "-"}\n匹配分: ${p.score || 0}\n说明: ${p.reason || "-"}`;
      editorUI.renderAiDetails(msg);
    });

    let editor = null;
    const editorUI = initEditorUI({
      onHeightChange: (h) => { if (editor) { editor.setHeight(h); persist(); refreshParcelMetricsFromMap(); } },
      onBaseChange: (b) => { if (editor) { editor.setBase(b); persist(); refreshParcelMetricsFromMap(); } },
      onFunctionChange: (f) => { if (editor) { editor.setFunction(f); persist(); refreshParcelMetricsFromMap(); } },
      onRotate: (deg) => { if (editor) { editor.rotateSelected(deg); persist(); } },
      onEconomicsChange: (p) => {
        if (editor) editor.setEconomicsParams(p);
        currentEconParams = JSON.parse(JSON.stringify(p));
        localStorage.setItem(ECON_PARAMS_CACHE_KEY, JSON.stringify(p));
        setStatus(`经济测算已更新：${new Date().toLocaleTimeString()}`);
      },
      onEconomicsRecalc: async (p) => {
        try {
          const economics = store.getStats(p || currentEconParams || {});
          const scenario = store.toScenarioJSON("scenario_mvp", p || currentEconParams || {});
          await saveEconomicsSnapshot({
            scenario_name: "scenario_mvp",
            economics_params: p || currentEconParams || {},
            economics,
            blocks_count: Number(economics?.blocks || 0),
            gfa: Number(economics?.gfa || 0),
            total_value: Number(economics?.total_value || 0),
            tdc: Number(economics?.tdc || 0),
            profit: Number(economics?.profit || 0),
            scenario,
          });
          setStatus(`重新计算并落盘成功：${new Date().toLocaleTimeString()}`);
        } catch (err) {
          setStatus(`重新计算已完成，但落盘失败：${err.message}`);
        }
      },
      onExport: () => {
        const payload = store.toScenarioJSON("scenario_mvp", currentEconParams);
        downloadJSON("scenario_mvp.json", payload);
      },
      onSaveLocal: () => { persist(); setStatus("Scenario saved to localStorage"); },
      onResetScenario: () => {
        if (!window.confirm("Reset scenario to initial state? This will clear unsaved edits.")) return;
        store.resetToInitial();
        refreshBuildingStackSource(map, { type: "FeatureCollection", features: [] });
        setLayerVisible(map, "buildings-stack-extrusion", false);
        setLayerVisible(map, "buildings-extrusion", true);
        localStorage.removeItem(SCENARIO_CACHE_KEY);
        selectedParcel = null;
        setSelectedParcel(map, null);
        document.getElementById("selected-parcel-id").textContent = "-";
        if (editor) editor.forceRefresh();
        setStatus("Scenario reset to initial state");
      },
      onDeleteSelected: () => { if (editor) { editor.deleteSelected(); persist(); refreshParcelMetricsFromMap(); } },
      onModeChange: (m) => editor && editor.setMode(m),
      onGenerateCluster: async () => {
        if (!selectedParcel) {
          setStatus("请先点击 zone-parcels 地块再生成。");
          return;
        }
        const payload = {
          scenario_name: "scenario_mvp",
          seed: Math.round(numOr(document.getElementById("gen-seed").value, 42)),
          template_id: document.getElementById("gen-template").value,
          zone_id: selectedParcel?.properties?.zone_id || document.getElementById("gen-zone").value,
          site_geojson: datasets.siteBoundary,
          zone_geojson: { type: "FeatureCollection", features: [selectedParcel] },
          constraints: {},
          intensity: {
            count: Math.max(1, Math.round(numOr(document.getElementById("gen-count").value, 12))),
            height_m: Math.max(3, numOr(document.getElementById("gen-height").value, 42)),
            functional_program: document.getElementById("gen-program").value,
          },
        };
        setStatus("Generating cluster in selected zone parcel...");
        try {
          const out = await requestClusterGeneration(payload);
          store.importGeneratedBlocks(out.blocks || []);
          if (editor) editor.forceRefresh();
          persist();
          refreshParcelMetricsFromMap();
          setStatus(`Generated ${out.diagnostics?.accepted || 0} blocks`);
        } catch (err) {
          setStatus(`Generate failed: ${err.message}`);
        }
      },
      onAiAllocate: async (prompt) => {
        const text = String(prompt || "").trim();
        if (!text) {
          editorUI.renderAiSummary("请输入一句愿景描述后再生成分层排布。");
          editorUI.renderAiDetails("");
          return;
        }
        try {
          const audienceSelect = document.getElementById("ai-audience-select");
          const audienceOther = document.getElementById("ai-audience-other");
          let audienceProfile = "";
          let audienceSource = "user";
          if (audienceSelect?.value === "待定") {
            setStatus("AI思考中：正在补全服务人群...");
            const completed = await requestAudienceCompletion(text);
            audienceProfile = (completed.audiences || []).join("、");
            audienceSource = completed.source || "agent";
          } else if (audienceSelect?.value === "其它") {
            audienceProfile = String(audienceOther?.value || "").trim() || "待定";
          } else {
            audienceProfile = String(audienceSelect?.value || "").trim();
          }

          const src = map.getSource("zone-parcels");
          const parcels = src && src._data ? src._data : datasets.zoneParcels;
          const current = store.getFeatureCollection();
          const publicFc = {
            type: "FeatureCollection",
            features: (current.features || []).filter((f) => String(f?.properties?.functionType || "").toUpperCase() === "PUBLIC"),
          };
          let agentPlan = null;
          if ((publicFc.features || []).length) {
            setStatus("AI思考中：正在细分PUBLIC功能...");
            try {
              agentPlan = await requestFloorPlanWithAgent({ prompt: text, blocksFc: publicFc, parcelsFc: parcels, audienceProfile });
            } catch {
              agentPlan = null;
            }
          }
          const publicIds = [];
          const lines = [];
          (current.features || []).forEach((f, idx) => {
            const code = String(f?.properties?.functionType || "").toUpperCase();
            if (code !== "PUBLIC") return;
            const id = String(f?.properties?._scenarioId || f?.properties?.id || `public_${idx + 1}`);
            const zone = String(f?.properties?.zone || "UNKNOWN");
            const fromAgent = (agentPlan?.outputs || []).find((x) => String(x.id) === id);
            const subtype = fromAgent?.segments?.[0]?.primary || inferPublicSubtype(text, zone);
            const item = store.getById(id);
            if (item) item.feature.properties.public_subtype = subtype;
            publicIds.push(id);
            lines.push(`- ${id}: PUBLIC -> ${subtype}`);
          });
          persist();
          if (editor) editor.forceRefresh();
          const summaryText = publicIds.length
            ? `已完成PUBLIC细分，共${publicIds.length}个体块 | 人群: ${audienceProfile || "未指定"} (${audienceSource})`
            : "未检测到PUBLIC体块，本次未执行细分。";
          editorUI.renderAiSummary(summaryText);
          editorUI.renderAiDetails(lines.join("\n"));
          zoneOverlay.render([], parcels);
          setStatus(publicIds.length ? `PUBLIC细分完成: ${publicIds.length} blocks` : "没有 PUBLIC 体块可细分");
        } catch (err) {
          setStatus(`AI调用失败：${err.message}`);
        }
      },
    });

    const lockBtn = document.getElementById("btn-toggle-lock");
    const setLockBtnText = () => {
      if (!editor) return;
      lockBtn.textContent = editor.isLocked() ? "Unlock Edit" : "Lock Edit";
    };

    const loadRhino = async (bootstrap = null) => {
      try {
        setStatus("Loading Rhino scenario...");
        let scenario;
        if (bootstrap?.scenario) {
          scenario = bootstrap.scenario;
          lastRhinoUpdatedAt = bootstrap.updatedAt || lastRhinoUpdatedAt;
        } else {
          const fetched = await fetchRhinoScenario();
          scenario = fetched.scenario;
          lastRhinoUpdatedAt = fetched.updatedAt || lastRhinoUpdatedAt;
        }
        const [rhinoParcels, rhinoOriginal, rhinoWalking, rhinoGround] = await Promise.all([
          fetchRhinoParcels(),
          fetchRhinoOriginalBuildings(),
          fetchRhinoWalking(),
          fetchRhinoGround(),
        ]);
        const split = (layer) => ({ type: "FeatureCollection", features: (rhinoParcels.features || []).filter((f) => String(f?.properties?.layer || "").toUpperCase() === layer) });
        const srcParcels = map.getSource("zone-parcels");
        if (srcParcels) srcParcels.setData(rhinoParcels);
        const srcCBD = map.getSource("zone-cbd"); if (srcCBD) srcCBD.setData(split("Z_CBD"));
        const srcTOD = map.getSource("zone-tod"); if (srcTOD) srcTOD.setData(split("Z_TOD"));
        const srcOFC = map.getSource("zone-ofc"); if (srcOFC) srcOFC.setData(split("Z_OFC"));
        const srcRES = map.getSource("zone-res"); if (srcRES) srcRES.setData(split("Z_RES"));
        const srcOriginal = map.getSource("rhino-original-buildings"); if (srcOriginal) srcOriginal.setData(rhinoOriginal);
        const srcWalking = map.getSource("rhino-walking"); if (srcWalking) srcWalking.setData(rhinoWalking);
        const srcGround = map.getSource("rhino-ground"); if (srcGround) srcGround.setData(rhinoGround);
        if (!store.loadScenarioJSON(scenario)) {
          setStatus("Rhino scenario invalid");
          return false;
        }
        refreshBuildingStackSource(map, { type: "FeatureCollection", features: [] });
        setLayerVisible(map, "buildings-stack-extrusion", false);
        setLayerVisible(map, "buildings-extrusion", true);
        if (editor) {
          editor.setLocked(true);
          editor.forceRefresh();
        }
        renderParcelMetrics(rhinoParcels, store.getFeatureCollection());
        const { bounds, count } = collectLngLatBoundsFromFeatureCollection(store.getFeatureCollection());
        if (count > 0 && !bounds.isEmpty()) {
          const sw = bounds.getSouthWest();
          const ne = bounds.getNorthEast();
          const validWgs84 = Math.abs(sw.lng) <= 180 && Math.abs(ne.lng) <= 180 && Math.abs(sw.lat) <= 90 && Math.abs(ne.lat) <= 90;
          if (validWgs84) {
            map.fitBounds(bounds, { padding: { top: 80, left: 380, right: 420, bottom: 80 }, duration: 500 });
          } else {
            setStatus("Rhino loaded, but coordinates are not WGS84 lon/lat. Please export CRS84/WGS84.");
            persist();
            setLockBtnText();
            return false;
          }
        }
        persist();
        setLockBtnText();
        setStatus(`Rhino loaded: ${(scenario.blocks || []).length} blocks | ground ${(rhinoGround?.features || []).length} features (locked)`);
        return true;
      } catch (err) {
        setStatus(`Load Rhino failed: ${err.message}`);
        return false;
      }
    };
    document.getElementById("btn-load-rhino").addEventListener("click", () => loadRhino());
    const hero = document.getElementById("btn-load-rhino-hero");
    if (hero) hero.addEventListener("click", () => loadRhino());

    (async () => {
      let rhinoLoaded = false;
      if (rhinoBootstrap) {
        rhinoLoaded = await loadRhino(rhinoBootstrap);
        if (!rhinoLoaded) {
          const cached = localStorage.getItem(SCENARIO_CACHE_KEY);
          if (cached) {
            try {
              const parsed = JSON.parse(cached);
              if (store.loadScenarioJSON(parsed)) setStatus("Rhino load failed; recovered last scenario from local cache");
            } catch {
              // ignore cache parse errors
            }
          }
        }
      }
      editor = createEditorController(map, store, editorUI);
      if (rhinoLoaded) {
        editor.setLocked(true);
      }
      if (cachedEconParams) {
        editorUI.setEconomicsInputs(cachedEconParams);
      }
      editorUI.pushEcoParams();
      setLockBtnText();
      if (!rhinoLoaded) {
        persist();
        refreshParcelMetricsFromMap();
      }
      window.addEventListener("beforeunload", persist);
      setStatus("Ready");
      setTimeout(() => { document.getElementById("status").style.display = "none"; }, 1200);
    })().catch((err) => {
      console.error(err);
      setStatus(`Failed: ${err.message}`);
    });

    lockBtn.addEventListener("click", () => {
      if (!editor) return;
      editor.setLocked(!editor.isLocked());
      setLockBtnText();
      setStatus(editor.isLocked() ? "Editor locked (Rhino-driven mode)" : "Editor unlocked");
    });

    document.addEventListener("keydown", (evt) => {
      const tag = (evt.target?.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) return;
      if (evt.key === "Delete") { editor.deleteSelected(); persist(); evt.preventDefault(); return; }
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "z") {
        if (evt.shiftKey) editor.redo(); else editor.undo();
        persist();
        evt.preventDefault();
      }
    });
  });
}

main().catch((err) => {
  console.error(err);
  setStatus(`Failed: ${err.message}`);
});
