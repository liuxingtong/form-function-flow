const DATASETS = {
  siteBoundary: "/data/site_dxf/crs84/SITE.geojson",
  buildings: "/data/site_design_platform/datasets/site_buildings.geojson",
  landuse: "/data/site_design_platform/datasets/site_landuse.geojson",
  summary: "/data/site_design_platform/datasets/summary.json",
  zCBD: "/data/district_dxf/crs84/Z_CBD.geojson",
  zTOD: "/data/district_dxf/crs84/Z_TOD.geojson",
  zOFC: "/data/district_dxf/crs84/Z_OFC.geojson",
};

const LANDUSE_COLORS = {
  "居住用地": "#f4a261",
  "商业服务用地": "#e76f51",
  "商务办公用地": "#b56576",
  "工业用地": "#6d597a",
  "行政办公用地": "#457b9d",
  "医疗卫生用地": "#2a9d8f",
  "交通场站用地": "#4d4d4d",
  "公园与绿地用地": "#7cb342",
};

const HEIGHT_BINS = [
  { label: "< 12m", color: "#fff4cc", max: 12 },
  { label: "12-24m", color: "#fdd870", max: 24 },
  { label: "24-36m", color: "#fca311", max: 36 },
  { label: "36-60m", color: "#f77f00", max: 60 },
  { label: "60-90m", color: "#d62828", max: 90 },
  { label: "> 90m", color: "#6a040f", max: 1e9 },
];

const DISTRICT_STYLE = {
  zCBD: { color: "#2f6df6", id: "zone-cbd" },
  zTOD: { color: "#17bebb", id: "zone-tod" },
  zOFC: { color: "#f7b32b", id: "zone-ofc" },
};

const statusEl = document.getElementById("status");
function setStatus(text) { statusEl.textContent = text; }

function addLegend(containerId, items) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "legend-item";
    row.innerHTML = `<span class="swatch" style="background:${item.color}"></span><span>${item.label}</span>`;
    el.appendChild(row);
  });
}

function heightColorExpression() {
  const expr = ["case"];
  HEIGHT_BINS.forEach((b) => {
    expr.push(["<=", ["to-number", ["get", "Height"], 0], b.max], b.color);
  });
  expr.push("#5c6b82");
  return expr;
}

function landuseColorExpression() {
  const expr = ["match", ["get", "class2"]];
  Object.entries(LANDUSE_COLORS).forEach(([k, c]) => expr.push(k, c));
  expr.push("#9aa7bc");
  return expr;
}

async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`${url} -> ${r.status}`);
  }
  return r.json();
}

function setLayerVisible(map, id, on) {
  if (map.getLayer(id)) {
    map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  }
}

function setLayerOpacity(map, id, value, key) {
  if (map.getLayer(id)) {
    map.setPaintProperty(id, key, value);
  }
}

function bindToggles(map) {
  const checks = [
    { id: "lyr-site-boundary", layers: ["site-boundary-line"] },
    { id: "lyr-buildings", layers: ["buildings-extrusion"] },
    { id: "lyr-landuse", layers: ["landuse-fill"] },
    { id: "lyr-zone-cbd", layers: ["zone-cbd-fill", "zone-cbd-line"] },
    { id: "lyr-zone-tod", layers: ["zone-tod-fill", "zone-tod-line"] },
    { id: "lyr-zone-ofc", layers: ["zone-ofc-fill", "zone-ofc-line"] },
  ];
  checks.forEach((cfg) => {
    const el = document.getElementById(cfg.id);
    el.addEventListener("change", () => cfg.layers.forEach((x) => setLayerVisible(map, x, el.checked)));
  });

  const sliders = [
    { id: "opa-zone-cbd", layer: "zone-cbd-fill" },
    { id: "opa-zone-tod", layer: "zone-tod-fill" },
    { id: "opa-zone-ofc", layer: "zone-ofc-fill" },
  ];
  sliders.forEach((cfg) => {
    const el = document.getElementById(cfg.id);
    el.addEventListener("input", () => setLayerOpacity(map, cfg.layer, Number(el.value) / 100, "fill-opacity"));
  });
}

function bindPopups(map) {
  map.on("click", "buildings-extrusion", (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: false })
      .setLngLat(e.lngLat)
      .setHTML(`<div><b>建筑</b><br/>ID: ${p.id}<br/>Height: ${Number(p.Height).toFixed(2)} m</div>`)
      .addTo(map);
  });
  map.on("click", "landuse-fill", (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: false })
      .setLngLat(e.lngLat)
      .setHTML(`<div><b>用地</b><br/>ID: ${p.id}<br/>Class2: ${p.class2}</div>`)
      .addTo(map);
  });
  ["buildings-extrusion", "landuse-fill"].forEach((id) => {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  });
}

async function main() {
  addLegend("height-legend", HEIGHT_BINS.map((x) => ({ label: x.label, color: x.color })));
  addLegend("landuse-legend", Object.entries(LANDUSE_COLORS).map(([label, color]) => ({ label, color })));

  const map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        carto: {
          type: "raster",
          tiles: [
            "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
          ],
          tileSize: 256,
          attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        },
      },
      layers: [{ id: "carto", type: "raster", source: "carto" }],
    },
    center: [121.4596, 31.2495],
    zoom: 15.2,
    pitch: 44,
    bearing: -18,
    minZoom: 11,
    maxZoom: 20,
  });

  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

  setStatus("Loading GeoJSON layers...");
  const [siteBoundary, buildings, landuse, zCBD, zTOD, zOFC, summary] = await Promise.all([
    loadJSON(DATASETS.siteBoundary),
    loadJSON(DATASETS.buildings),
    loadJSON(DATASETS.landuse),
    loadJSON(DATASETS.zCBD),
    loadJSON(DATASETS.zTOD),
    loadJSON(DATASETS.zOFC),
    loadJSON(DATASETS.summary),
  ]);

  document.getElementById("buildings-count").textContent = `${summary.buildings_features} features`;
  document.getElementById("landuse-count").textContent = `${summary.landuse_features} features`;
  document.getElementById("summary").innerHTML =
    `<div><span class="badge">建筑数:</span> ${summary.buildings_features}</div>` +
    `<div><span class="badge">用地数:</span> ${summary.landuse_features}</div>` +
    `<div><span class="badge">高度范围:</span> ${summary.height_min.toFixed(1)}m - ${summary.height_max.toFixed(1)}m</div>`;

  map.on("load", () => {
    map.addSource("site-boundary", { type: "geojson", data: siteBoundary });
    map.addSource("buildings", { type: "geojson", data: buildings });
    map.addSource("landuse", { type: "geojson", data: landuse });
    map.addSource("zone-cbd", { type: "geojson", data: zCBD });
    map.addSource("zone-tod", { type: "geojson", data: zTOD });
    map.addSource("zone-ofc", { type: "geojson", data: zOFC });

    map.addLayer({
      id: "landuse-fill",
      type: "fill",
      source: "landuse",
      paint: {
        "fill-color": landuseColorExpression(),
        "fill-opacity": 0.44,
        "fill-outline-color": "rgba(255,255,255,0.28)",
      },
    });

    map.addLayer({
      id: "buildings-extrusion",
      type: "fill-extrusion",
      source: "buildings",
      paint: {
        "fill-extrusion-color": heightColorExpression(),
        "fill-extrusion-height": ["to-number", ["get", "Height"], 8],
        "fill-extrusion-base": 0,
        "fill-extrusion-opacity": 0.87,
      },
    });

    Object.values(DISTRICT_STYLE).forEach((st) => {
      map.addLayer({
        id: `${st.id}-fill`,
        type: "fill",
        source: st.id,
        paint: { "fill-color": st.color, "fill-opacity": 0.45 },
      });
      map.addLayer({
        id: `${st.id}-line`,
        type: "line",
        source: st.id,
        paint: { "line-color": st.color, "line-width": 1.6, "line-opacity": 0.92 },
      });
    });

    map.addLayer({
      id: "site-boundary-line",
      type: "line",
      source: "site-boundary",
      paint: { "line-color": "#101521", "line-width": 3, "line-opacity": 0.95 },
    });

    const bounds = new maplibregl.LngLatBounds();
    siteBoundary.features.forEach((f) => {
      const g = f.geometry;
      const sets = g.type === "Polygon" ? g.coordinates : (g.type === "LineString" ? [g.coordinates] : []);
      sets.forEach((ring) => ring.forEach((c) => bounds.extend(c)));
    });
    if (!bounds.isEmpty()) {
      map.fitBounds(bounds, { padding: { top: 80, left: 380, right: 80, bottom: 80 }, duration: 600 });
    }

    bindToggles(map);
    bindPopups(map);
    setStatus("Ready");
    setTimeout(() => { statusEl.style.display = "none"; }, 1200);
  });
}

main().catch((err) => {
  console.error(err);
  setStatus("Failed: " + err.message);
});

