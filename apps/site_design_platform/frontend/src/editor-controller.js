import { refreshBuildingsSource, setSelectedBuildingStyle, setVertexMarkers } from "./layer-manager.js";

function nearestVertexIdx(feature, lngLat) {
  const ring = feature?.geometry?.type === "Polygon" ? feature.geometry.coordinates[0] : null;
  if (!ring || ring.length < 4) return -1;
  let best = -1;
  let minD = Infinity;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const d = Math.hypot(ring[i][0] - lngLat.lng, ring[i][1] - lngLat.lat);
    if (d < minD) { minD = d; best = i; }
  }
  return best;
}

export function createEditorController(map, store, ui) {
  const selectedIds = new Set();
  let activeId = null;
  let dragging = false;
  let vertexDragging = false;
  let dragStart = null;
  let mode = "SELECT";
  let activeVertexIdx = -1;
  let locked = false;
  const econParams = { years: 5 };
  const selectedArray = () => Array.from(selectedIds);

  const canvas = map.getCanvasContainer();
  let box = null;
  let boxStart = null;

  function getSelection() {
    const ids = selectedArray();
    if (ids.length === 0) return null;
    if (ids.length > 1) return { scenarioId: `${ids.length} selected`, multi: true };
    const item = store.getById(ids[0]);
    if (!item) return null;
    return { scenarioId: ids[0], height: item.metrics.height, base: item.metrics.base, functionType: item.feature.properties.functionType || "MIXED_USE", multi: false };
  }

  const refresh = () => {
    refreshBuildingsSource(map, store.getFeatureCollection());
    setSelectedBuildingStyle(map, selectedArray());
    const vertexMarkers = mode === "VERTEX" && activeId && selectedIds.size === 1 ? store.getVertexMarkers(activeId) : { type: "FeatureCollection", features: [] };
    setVertexMarkers(map, vertexMarkers);
    ui.renderSelection(getSelection());
    ui.renderDashboard(store.getStats(econParams));
    ui.renderMode(mode);
  };

  function setSingleSelection(id) {
    selectedIds.clear();
    if (id) selectedIds.add(id);
    activeId = id;
    activeVertexIdx = -1;
  }

  function toggleSelection(id) {
    if (!id) return;
    if (selectedIds.has(id)) selectedIds.delete(id); else selectedIds.add(id);
    activeId = selectedIds.has(id) ? id : (selectedArray()[0] || null);
    activeVertexIdx = -1;
  }

  function addSelectionIds(ids) {
    ids.forEach((id) => selectedIds.add(id));
    activeId = selectedArray()[0] || null;
  }

  function startBoxSelect(e) {
    boxStart = e.point;
    box = document.createElement("div");
    box.style.position = "absolute";
    box.style.border = "1px dashed #7cc4ff";
    box.style.background = "rgba(124,196,255,0.15)";
    box.style.pointerEvents = "none";
    canvas.appendChild(box);
  }

  function updateBoxSelect(point) {
    if (!box || !boxStart) return;
    const minX = Math.min(boxStart.x, point.x);
    const minY = Math.min(boxStart.y, point.y);
    const maxX = Math.max(boxStart.x, point.x);
    const maxY = Math.max(boxStart.y, point.y);
    box.style.left = `${minX}px`;
    box.style.top = `${minY}px`;
    box.style.width = `${maxX - minX}px`;
    box.style.height = `${maxY - minY}px`;
  }

  function finishBoxSelect(point) {
    if (!boxStart) return;
    const min = [Math.min(boxStart.x, point.x), Math.min(boxStart.y, point.y)];
    const max = [Math.max(boxStart.x, point.x), Math.max(boxStart.y, point.y)];
    const features = map.queryRenderedFeatures([min, max], { layers: ["buildings-extrusion"] });
    addSelectionIds(features.map((f) => f.properties?._scenarioId).filter(Boolean));
    if (box && box.parentNode) box.parentNode.removeChild(box);
    box = null;
    boxStart = null;
    refresh();
  }

  map.on("click", "buildings-extrusion", (e) => {
    if (locked) return;
    const id = e.features?.[0]?.properties?._scenarioId || null;
    if (!id || mode === "ADD") return;
    if (e.originalEvent?.shiftKey) toggleSelection(id); else setSingleSelection(id);
    if (mode === "VERTEX" && activeId && selectedIds.size === 1) {
      const item = store.getById(activeId);
      activeVertexIdx = nearestVertexIdx(item?.feature, e.lngLat);
    }
    refresh();
  });

  map.on("click", (e) => {
    if (locked) return;
    if (mode === "ADD") {
      const id = store.addBlockAt(e.lngLat.lng, e.lngLat.lat);
      setSingleSelection(id);
      refresh();
      return;
    }
    const hits = map.queryRenderedFeatures(e.point, { layers: ["buildings-extrusion"] });
    if (!hits.length && !e.originalEvent?.shiftKey) {
      setSingleSelection(null);
      refresh();
    }
  });

  map.on("mousedown", (e) => {
    if (locked) return;
    if (mode !== "SELECT") return;
    if (!e.originalEvent?.shiftKey) return;
    const hits = map.queryRenderedFeatures(e.point, { layers: ["buildings-extrusion"] });
    if (hits.length) return;
    startBoxSelect(e);
    e.preventDefault();
  });

  map.on("mousedown", "buildings-extrusion", (e) => {
    if (locked) return;
    const clickedId = e.features?.[0]?.properties?._scenarioId;
    if (!clickedId) return;
    if (mode === "SELECT") {
      if (!selectedIds.has(clickedId)) return;
      dragging = true;
      dragStart = e.lngLat;
      map.getCanvas().style.cursor = "grabbing";
      e.preventDefault();
    } else if (mode === "VERTEX" && selectedIds.size === 1 && clickedId === activeId) {
      const item = store.getById(activeId);
      activeVertexIdx = nearestVertexIdx(item?.feature, e.lngLat);
      if (activeVertexIdx >= 0) {
        vertexDragging = true;
        map.getCanvas().style.cursor = "crosshair";
      }
      e.preventDefault();
    }
  });

  map.on("mousemove", (e) => {
    if (locked) return;
    if (boxStart && box) {
      updateBoxSelect(e.point);
      return;
    }
    if (dragging && dragStart && selectedIds.size > 0) {
      const dLng = e.lngLat.lng - dragStart.lng;
      const dLat = e.lngLat.lat - dragStart.lat;
      const metersLng = dLng * 111320 * Math.cos((dragStart.lat * Math.PI) / 180);
      const metersLat = dLat * 111320;
      store.translate(selectedArray(), metersLng, metersLat, dragStart.lat, false);
      dragStart = e.lngLat;
      refresh();
      return;
    }
    if (vertexDragging && activeId && activeVertexIdx >= 0) {
      store.updateVertex(activeId, activeVertexIdx, e.lngLat.lng, e.lngLat.lat, { snapDeg: 0.00002, minEdgeDeg: 0.00002, withCheckpoint: false });
      refresh();
    }
  });

  map.on("mouseup", (e) => {
    if (boxStart && box) {
      finishBoxSelect(e.point);
      return;
    }
    dragging = false;
    vertexDragging = false;
    dragStart = null;
    map.getCanvas().style.cursor = "";
  });

  refresh();

  return {
    forceRefresh() { refresh(); },
    setHeight(h) { if (!locked && selectedIds.size) { store.setHeight(selectedArray(), h); refresh(); } },
    setBase(b) { if (!locked && selectedIds.size) { store.setBase(selectedArray(), b); refresh(); } },
    setFunction(f) { if (!locked && selectedIds.size) { store.setFunction(selectedArray(), f); refresh(); } },
    rotateSelected(deg) { if (!locked && selectedIds.size) { store.rotate(selectedArray(), deg); refresh(); } },
    setEconomicsParams(p) { Object.assign(econParams, p); refresh(); },
    setMode(nextMode) { if (!locked) { mode = nextMode; refresh(); } },
    deleteSelected() { if (!locked && selectedIds.size) { store.deleteBlocks(selectedArray()); selectedIds.clear(); activeId = null; activeVertexIdx = -1; refresh(); } },
    undo() { if (!locked && store.undo()) refresh(); },
    redo() { if (!locked && store.redo()) refresh(); },
    setLocked(v) { locked = !!v; if (locked) { dragging = false; vertexDragging = false; boxStart = null; if (box && box.parentNode) box.parentNode.removeChild(box); box = null; } refresh(); },
    isLocked() { return locked; },
  };
}
