import { DISTRICT_STYLE, FUNCTION_COLORS, LANDUSE_COLORS } from "./config.js";

function landuseColorExpression() {
  const expr = ["match", ["get", "class2"]];
  Object.entries(LANDUSE_COLORS).forEach(([k, c]) => expr.push(k, c));
  expr.push("#9aa7bc");
  return expr;
}

function functionColorExpression() {
  const expr = ["match", ["get", "functionType"]];
  Object.entries(FUNCTION_COLORS).forEach(([k, c]) => expr.push(k, c));
  expr.push("#7d8ca3");
  return expr;
}

export function addSourcesAndLayers(map, datasets) {
  map.addSource("site-boundary", { type: "geojson", data: datasets.siteBoundary });
  map.addSource("buildings", { type: "geojson", data: datasets.buildings });
  map.addSource("buildings-stack", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("landuse", { type: "geojson", data: datasets.landuse });
  map.addSource("zone-parcels", { type: "geojson", data: datasets.zoneParcels });
  map.addSource("zone-cbd", { type: "geojson", data: datasets.zCBD });
  map.addSource("zone-tod", { type: "geojson", data: datasets.zTOD });
  map.addSource("zone-ofc", { type: "geojson", data: datasets.zOFC });
  map.addSource("zone-res", { type: "geojson", data: datasets.zRES });
  map.addSource("rhino-original-buildings", { type: "geojson", data: datasets.rhinoOriginalBuildings || { type: "FeatureCollection", features: [] } });
  map.addSource("rhino-walking", { type: "geojson", data: datasets.rhinoWalking || { type: "FeatureCollection", features: [] } });
  map.addSource("rhino-ground", { type: "geojson", data: datasets.rhinoGround || { type: "FeatureCollection", features: [] } });
  map.addSource("editor-vertices", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("selected-parcel", { type: "geojson", data: { type: "FeatureCollection", features: [] } });

  map.addLayer({ id: "landuse-fill", type: "fill", source: "landuse", paint: { "fill-color": landuseColorExpression(), "fill-opacity": 0.34, "fill-outline-color": "rgba(255,255,255,0.22)" } });

  map.addLayer({ id: "zone-parcels-fill", type: "fill", source: "zone-parcels", paint: { "fill-color": "rgba(0,0,0,0)", "fill-opacity": 0.01 } });
  map.addLayer({ id: "zone-parcels-line", type: "line", source: "zone-parcels", paint: { "line-color": "#59aaff", "line-width": 2.0, "line-opacity": 0.9 } });

  map.addLayer({ id: "selected-parcel-fill", type: "fill", source: "selected-parcel", paint: { "fill-color": "#00d4ff", "fill-opacity": 0.22 } });
  map.addLayer({ id: "selected-parcel-line", type: "line", source: "selected-parcel", paint: { "line-color": "#00f0ff", "line-width": 5.0, "line-opacity": 1.0 } });

  map.addLayer({ id: "buildings-extrusion", type: "fill-extrusion", source: "buildings", paint: { "fill-extrusion-color": functionColorExpression(), "fill-extrusion-height": ["+", ["to-number", ["get", "Base"], 0], ["to-number", ["get", "Height"], 8]], "fill-extrusion-base": ["to-number", ["get", "Base"], 0], "fill-extrusion-opacity": 0.87 } });
  map.addLayer({ id: "buildings-stack-extrusion", type: "fill-extrusion", source: "buildings-stack", paint: { "fill-extrusion-color": ["match", ["get", "functionCode"], "CENTER_OFFICE", "#0b3c9d", "SMALL_OFFICE", "#2f80ed", "APARTMENT", "#2e7d32", "HOTEL", "#e53935", "RESIDENTIAL", "#7cb342", "CENTER_COMMERCIAL", "#fb8c00", "LEISURE_COMMERCIAL", "#fdd835", "PUBLIC", "#00897b", "GREEN", "#26a69a", "GROUND", "#bfc5cc", "#7d8ca3"], "fill-extrusion-height": ["+", ["to-number", ["get", "Base"], 0], ["to-number", ["get", "Height"], 0]], "fill-extrusion-base": ["to-number", ["get", "Base"], 0], "fill-extrusion-opacity": 0.84 } });
  map.addLayer({ id: "rhino-original-buildings-extrusion", type: "fill-extrusion", source: "rhino-original-buildings", paint: { "fill-extrusion-color": "#ffffff", "fill-extrusion-height": ["+", ["to-number", ["get", "Base"], 0], ["to-number", ["get", "Height"], 8]], "fill-extrusion-base": ["to-number", ["get", "Base"], 0], "fill-extrusion-opacity": 0.28 } });
  map.addLayer({ id: "rhino-walking-line", type: "line", source: "rhino-walking", paint: { "line-color": "#ffffff", "line-opacity": 0.7, "line-width": 2.0, "line-dasharray": [1.2, 1.2] } });
  map.addLayer({ id: "rhino-walking-fill", type: "fill", source: "rhino-walking", paint: { "fill-color": "#ffffff", "fill-opacity": 0.16 } });
  map.addLayer({ id: "rhino-ground-fill", type: "fill", source: "rhino-ground", paint: { "fill-color": "#c6ccd3", "fill-opacity": 0.36, "fill-outline-color": "rgba(166,175,186,0.9)" } });
  map.addLayer({ id: "rhino-ground-line", type: "line", source: "rhino-ground", paint: { "line-color": "#9aa3ad", "line-opacity": 0.95, "line-width": 1.6 } });

  Object.values(DISTRICT_STYLE).forEach((st) => {
    map.addLayer({ id: `${st.id}-fill`, type: "fill", source: st.id, paint: { "fill-color": st.color, "fill-opacity": 0.26 } });
    map.addLayer({ id: `${st.id}-line`, type: "line", source: st.id, paint: { "line-color": st.color, "line-width": 1.6, "line-opacity": 0.88 } });
  });

  map.addLayer({ id: "site-boundary-line", type: "line", source: "site-boundary", paint: { "line-color": "#101521", "line-width": 3, "line-opacity": 0.95 } });
  map.addLayer({ id: "buildings-selected-line", type: "line", source: "buildings", filter: ["==", ["get", "_scenarioId"], "__none__"], paint: { "line-color": "#36c5f0", "line-width": 3.2, "line-opacity": 0.95 } });
  map.addLayer({ id: "editor-vertices-circle", type: "circle", source: "editor-vertices", paint: { "circle-radius": 5, "circle-color": "#ffffff", "circle-stroke-color": "#36c5f0", "circle-stroke-width": 2 } });
}

export function fitToSiteBoundary(map, siteBoundary) {
  const bounds = new maplibregl.LngLatBounds();
  siteBoundary.features.forEach((f) => {
    const g = f.geometry;
    const sets = g.type === "Polygon" ? g.coordinates : (g.type === "LineString" ? [g.coordinates] : []);
    sets.forEach((ring) => ring.forEach((c) => bounds.extend(c)));
  });
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: { top: 80, left: 380, right: 420, bottom: 80 }, duration: 600 });
}

export function refreshBuildingsSource(map, buildingsGeojson) { const src = map.getSource("buildings"); if (src) src.setData(buildingsGeojson); }
export function refreshBuildingStackSource(map, stackGeojson) { const src = map.getSource("buildings-stack"); if (src) src.setData(stackGeojson); }
export function setVertexMarkers(map, featureCollection) { const src = map.getSource("editor-vertices"); if (src) src.setData(featureCollection); }
export function setSelectedParcel(map, feature) { const src = map.getSource("selected-parcel"); if (src) src.setData({ type: "FeatureCollection", features: feature ? [feature] : [] }); }

export function setSelectedBuildingStyle(map, scenarioIds) {
  if (!map.getLayer("buildings-selected-line")) return;
  const ids = scenarioIds || [];
  if (!ids.length) {
    map.setFilter("buildings-selected-line", ["==", ["get", "_scenarioId"], "__none__"]);
    return;
  }
  const expr = ["any", ...ids.map((id) => ["==", ["get", "_scenarioId"], id])];
  map.setFilter("buildings-selected-line", expr);
}

export function setLayerVisible(map, id, on) { if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); }
export function setLayerOpacity(map, id, value, key) { if (map.getLayer(id)) map.setPaintProperty(id, key, value); }
