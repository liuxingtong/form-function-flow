const ZONE_TITLE = {
  CBD: "CBD",
  LEISURE: "休闲商业区",
  RESIDENTIAL: "居住区",
  CREATIVE: "文创区",
  UNKNOWN: "未识别分区",
};

const FUNC_COLOR = {
  CENTER_OFFICE: "#0b3c9d",
  SMALL_OFFICE: "#2f80ed",
  HOTEL: "#e53935",
  RESIDENTIAL: "#66bb6a",
  CENTER_COMMERCIAL: "#ff8f00",
  LEISURE_COMMERCIAL: "#ffca28",
  PUBLIC: "#00bcd4",
  GREEN: "#26a69a",
  COVER_MASS: "#8d6e63",
  WALKWAY: "#9e9e9e",
  HIGHWAY: "#424242",
  GROUND: "#bfc5cc",
};

const FUNC_LABEL = {
  CENTER_OFFICE: "总部办公",
  SMALL_OFFICE: "中小办公",
  HOTEL: "酒店",
  RESIDENTIAL: "居住",
  CENTER_COMMERCIAL: "商业",
  LEISURE_COMMERCIAL: "休闲",
  PUBLIC: "公共",
  GREEN: "绿地",
  COVER_MASS: "覆盖体量",
  WALKWAY: "慢行",
  HIGHWAY: "道路",
  GROUND: "地面",
};

function ring(feature) { return feature?.geometry?.type === "Polygon" ? (feature.geometry.coordinates?.[0] || []) : []; }
function centroid(feature) {
  const r = ring(feature);
  if (!r.length) return null;
  let x = 0; let y = 0; let n = 0;
  r.slice(0, -1).forEach((p) => { x += p[0]; y += p[1]; n += 1; });
  return n ? [x / n, y / n] : r[0];
}
function zoneCentroidsFromParcels(parcelsFc) {
  const grouped = {};
  (parcelsFc?.features || []).forEach((f) => {
    const layer = String(f?.properties?.layer || f?.properties?.zone_id || "").toUpperCase();
    let zone = null;
    if (layer === "Z_CBD") zone = "CBD";
    else if (layer === "Z_TOD") zone = "LEISURE";
    else if (layer === "Z_OFC") zone = "CREATIVE";
    else if (layer === "Z_RES") zone = "RESIDENTIAL";
    if (!zone) return;
    const c = centroid(f);
    if (!c) return;
    grouped[zone] = grouped[zone] || [];
    grouped[zone].push(c);
  });
  const out = {};
  Object.entries(grouped).forEach(([zone, arr]) => {
    out[zone] = [arr.reduce((a, p) => a + p[0], 0) / arr.length, arr.reduce((a, p) => a + p[1], 0) / arr.length];
  });
  return out;
}
function pieGradient(ratios) {
  let acc = 0;
  const parts = [];
  ratios.forEach((r) => {
    const color = FUNC_COLOR[r.key] || "#7d8ca3";
    const start = Math.round(acc * 360);
    acc += r.ratio;
    const end = Math.round(acc * 360);
    parts.push(`${color} ${start}deg ${end}deg`);
  });
  return `conic-gradient(${parts.join(",")})`;
}
function renderLegend(ratios) {
  return ratios.slice(0, 4).map((r) => `<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#dbe6fb;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${FUNC_COLOR[r.key] || "#7d8ca3"}"></span>${FUNC_LABEL[r.key] || r.key} ${(r.ratio * 100).toFixed(0)}%</div>`).join("");
}
function createPanelHtml(insight) {
  const title = `${insight.narrativeName || "功能复合区"}（${ZONE_TITLE[insight.zone] || insight.zone}）`;
  return `<div style="position:relative;width:230px;transform:translateY(-150px);background:rgba(18,24,35,.92);border:1px solid rgba(255,255,255,.18);border-radius:12px;padding:8px 10px;color:#eaf0fb;backdrop-filter:blur(4px);box-shadow:0 8px 24px rgba(0,0,0,.25)">
    <div style="font-size:12px;font-weight:700;margin-bottom:6px">${title}</div>
    <div style="display:flex;gap:10px;align-items:center">
      <div style="width:58px;height:58px;border-radius:50%;background:${pieGradient(insight.ratios || [])};border:1px solid rgba(255,255,255,.3)"></div>
      <div style="display:flex;flex-direction:column;gap:2px">${renderLegend(insight.ratios || [])}</div>
    </div>
    <div style="margin-top:6px;font-size:11px;line-height:1.35;color:#cfe0ff">${insight.headline || ""}</div>
    <div style="position:absolute;left:50%;bottom:-144px;height:136px;border-left:2px dashed rgba(190,210,245,.9);transform:translateX(-50%);"></div>
    <div style="position:absolute;left:50%;bottom:-152px;width:8px;height:8px;background:rgba(190,210,245,.95);border-radius:50%;transform:translateX(-50%);"></div>
  </div>`;
}

export function createZoneInsightOverlay(map) {
  const markers = [];
  function clear() { while (markers.length) markers.pop().remove(); }
  function render(zoneInsights, parcelsFc) {
    clear();
    if (!Array.isArray(zoneInsights) || !zoneInsights.length) return;
    const centers = zoneCentroidsFromParcels(parcelsFc);
    zoneInsights.forEach((z) => {
      const c = centers[z.zone];
      if (!c) return;
      const el = document.createElement("div");
      el.innerHTML = createPanelHtml(z);
      markers.push(new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat(c).addTo(map));
    });
  }
  return { render, clear };
}
