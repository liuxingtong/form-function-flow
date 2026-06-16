const KEYWORDS = {
  business: ["商务", "总部", "办公", "金融", "企业", "国际化"],
  leisure: ["休闲", "消费", "商业", "街区", "餐饮", "游逛", "夜游"],
  quiet: ["安静", "宜居", "居住", "社区", "家庭", "宁静"],
  creative: ["文创", "创意", "艺术", "设计", "展览", "文化"],
  night: ["夜间", "夜生活", "夜晚", "周末", "演艺", "剧场"],
  open: ["开放", "公共", "广场", "滨水", "慢行", "步行"],
  landmark: ["地标", "门户", "形象", "城市中心"],
};

const ZONE_CANDIDATES = {
  CBD: ["OFFICE", "CENTER_COMMERCIAL", "PUBLIC", "LEISURE_COMMERCIAL", "RESIDENTIAL"],
  LEISURE: ["LEISURE_COMMERCIAL", "CENTER_COMMERCIAL", "PUBLIC", "OFFICE", "RESIDENTIAL"],
  RESIDENTIAL: ["RESIDENTIAL", "PUBLIC", "CENTER_COMMERCIAL", "OFFICE"],
  CREATIVE: ["OFFICE", "PUBLIC", "LEISURE_COMMERCIAL", "CENTER_COMMERCIAL", "RESIDENTIAL"],
  UNKNOWN: ["OFFICE", "CENTER_COMMERCIAL", "RESIDENTIAL", "PUBLIC", "LEISURE_COMMERCIAL"],
};

function scoreText(text) {
  const t = String(text || "");
  const out = {
    business: 50, leisure: 50, quiet: 50, creative: 50, night: 50, open: 50, landmark: 50,
  };
  Object.entries(KEYWORDS).forEach(([k, words]) => {
    let hits = 0;
    words.forEach((w) => { if (t.includes(w)) hits += 1; });
    if (hits > 0) out[k] = Math.min(100, 50 + hits * 14);
  });
  return out;
}

function ringOfFeature(f) {
  if (!f || !f.geometry || f.geometry.type !== "Polygon") return [];
  return f.geometry.coordinates?.[0] || [];
}

function centroid(f) {
  const ring = ringOfFeature(f);
  if (!ring.length) return null;
  let x = 0; let y = 0; let n = 0;
  ring.slice(0, -1).forEach((p) => { x += p[0]; y += p[1]; n += 1; });
  return n ? [x / n, y / n] : ring[0];
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

function zoneRole(block, parcels) {
  const c = centroid(block);
  if (!c) return "UNKNOWN";
  let z = "UNKNOWN";
  (parcels?.features || []).forEach((p) => {
    const layer = String(p?.properties?.layer || p?.properties?.zone_id || "").toUpperCase();
    if (!layer.startsWith("Z_")) return;
    const ring = ringOfFeature(p);
    if (!ring.length) return;
    if (!pointInRing(c[0], c[1], ring)) return;
    if (layer === "Z_CBD") z = "CBD";
    else if (layer === "Z_TOD") z = "LEISURE";
    else if (layer === "Z_RES") z = "RESIDENTIAL";
    else if (layer === "Z_OFC") z = "CREATIVE";
  });
  return z;
}

function floorBand(block) {
  const h = Number(block?.properties?.Height ?? block?.properties?.height ?? 24);
  if (h <= 15) return "GROUND_LOW";
  if (h <= 42) return "MID";
  return "HIGH";
}

function featureId(block, idx) {
  return String(block?.properties?._scenarioId || block?.properties?.id || `blk_${idx + 1}`);
}

function pickFunction(zone, band, intent) {
  const candidates = ZONE_CANDIDATES[zone] || ZONE_CANDIDATES.UNKNOWN;
  const scored = candidates.map((fn) => {
    let s = 55;
    if (fn === "OFFICE") s += intent.business * 0.22 + intent.landmark * 0.08;
    if (fn === "CENTER_COMMERCIAL") s += intent.leisure * 0.2 + intent.open * 0.12;
    if (fn === "LEISURE_COMMERCIAL") s += intent.leisure * 0.24 + intent.night * 0.22;
    if (fn === "RESIDENTIAL") s += intent.quiet * 0.26;
    if (fn === "PUBLIC") s += intent.open * 0.2 + intent.creative * 0.12;

    if (zone === "CBD") {
      if (fn === "OFFICE") s += 18;
      if (fn === "CENTER_COMMERCIAL") s += 8;
      if (fn === "LEISURE_COMMERCIAL") s -= 6;
    }
    if (zone === "LEISURE") {
      if (fn === "LEISURE_COMMERCIAL") s += 14;
      if (fn === "CENTER_COMMERCIAL") s += 8;
      if (fn === "OFFICE") s -= 8;
    }
    if (zone === "RESIDENTIAL") {
      if (fn === "RESIDENTIAL") s += 20;
      if (fn === "PUBLIC") s += 8;
      if (fn === "LEISURE_COMMERCIAL") s -= 22;
    }
    if (zone === "CREATIVE") {
      if (fn === "PUBLIC") s += 10;
      if (fn === "OFFICE") s += 10;
      if (fn === "LEISURE_COMMERCIAL") s += 4;
    }

    if (zone === "RESIDENTIAL" && fn === "LEISURE_COMMERCIAL") s -= 80;
    if (zone === "RESIDENTIAL" && band === "HIGH" && (fn === "CENTER_COMMERCIAL" || fn === "LEISURE_COMMERCIAL")) s -= 90;
    if (zone === "LEISURE" && band === "GROUND_LOW" && fn === "OFFICE") s -= 35;
    if (zone === "CBD" && band === "HIGH" && fn === "PUBLIC") s -= 25;
    if (intent.quiet > 70 && zone === "RESIDENTIAL" && fn === "CENTER_COMMERCIAL") s -= 40;

    return { fn, score: s };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored[0].fn;
}

export function runFunctionalPlanning(prompt, blocksFc, parcelsFc) {
  const intent = scoreText(prompt);
  const assignments = [];
  (blocksFc?.features || []).forEach((b, idx) => {
    const zone = zoneRole(b, parcelsFc);
    const band = floorBand(b);
    const next = pickFunction(zone, band, intent);
    assignments.push({ id: featureId(b, idx), zone, band, functionType: next });
  });
  return {
    intent,
    assignments,
    summary: `AI分配完成：${assignments.length}个体块。愿景权重 -> 商务${intent.business}/休闲${intent.leisure}/居住安静${intent.quiet}/文创${intent.creative}/夜间${intent.night}/公共开放${intent.open}`,
  };
}
