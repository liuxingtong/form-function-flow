export function createMap() {
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
            "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
          ],
          tileSize: 256,
          attribution: "&copy; OpenStreetMap &copy; CARTO"
        }
      },
      layers: [{ id: "carto", type: "raster", source: "carto" }]
    },
    center: [121.4596, 31.2495],
    zoom: 15.2,
    pitch: 44,
    bearing: -18,
    minZoom: 11,
    maxZoom: 20
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  return map;
}
