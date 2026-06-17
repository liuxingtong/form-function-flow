import { FUNCTION_COLORS, FUNCTION_TYPES } from "./config.js";
import { createEconomicsUI } from "./economics-ui.js";
import { setLayerOpacity, setLayerVisible } from "./layer-manager.js";

export function setStatus(text) {
  const el = document.getElementById("status");
  if (!el) return;
  el.style.display = "block";
  el.textContent = text;
}

function addLegend(containerId, items) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "legend-item";
    row.innerHTML = `<span class="swatch" style="background:${item.color}"></span><span>${item.label}</span>`;
    el.appendChild(row);
  });
}

export function initLegends() {
  addLegend("function-legend", Object.entries(FUNCTION_COLORS).map(([label, color]) => ({ label, color })));
}

export function renderSummary(summary) {
  document.getElementById("buildings-count").textContent = `${summary.buildings_features} features`;
  const landuseCountEl = document.getElementById("landuse-count");
  if (landuseCountEl) landuseCountEl.textContent = `${summary.landuse_features} features`;
  document.getElementById("summary").innerHTML =
    `<div><span class="badge">Buildings</span> ${summary.buildings_features}</div>` +
    `<div><span class="badge">Landuse</span> ${summary.landuse_features}</div>` +
    `<div><span class="badge">Height Range:</span> ${summary.height_min.toFixed(1)}m - ${summary.height_max.toFixed(1)}m</div>`;
}

export function bindLayerToggles(map) {
  const checks = [
    { id: "lyr-site-boundary", layers: ["site-boundary-line"] },
    { id: "lyr-buildings", layers: ["buildings-extrusion", "buildings-stack-extrusion", "buildings-selected-line", "editor-vertices-circle"] },
    { id: "lyr-rhino-original", layers: ["rhino-original-buildings-extrusion"] },
    { id: "lyr-zone-cbd", layers: ["zone-cbd-fill", "zone-cbd-line"] },
    { id: "lyr-zone-tod", layers: ["zone-tod-fill", "zone-tod-line"] },
    { id: "lyr-zone-ofc", layers: ["zone-ofc-fill", "zone-ofc-line"] },
    { id: "lyr-zone-res", layers: ["zone-res-fill", "zone-res-line"] },
  ];
  checks.forEach((cfg) => {
    const el = document.getElementById(cfg.id);
    if (!el) return;
    const apply = () => cfg.layers.forEach((x) => setLayerVisible(map, x, el.checked));
    el.addEventListener("change", apply);
    apply();
  });

  ["cbd", "tod", "ofc", "res"].forEach((k) => {
    const el = document.getElementById(`opa-zone-${k}`);
    if (!el) return;
    el.addEventListener("input", () => setLayerOpacity(map, `zone-${k}-fill`, Number(el.value) / 100, "fill-opacity"));
  });
}

export function initEditorUI(handlers) {
  const numFrom = (id) => {
    const el = document.getElementById(id);
    return Number(el ? el.value : 0);
  };

  const fnSelect = document.getElementById("edit-function");
  FUNCTION_TYPES.forEach((name) => {
    const op = document.createElement("option");
    op.value = name;
    op.textContent = name;
    fnSelect.appendChild(op);
  });

  const hRange = document.getElementById("edit-height-range");
  const hInput = document.getElementById("edit-height-input");
  const bRange = document.getElementById("edit-base-range");
  const bInput = document.getElementById("edit-base-input");
  const rotateInput = document.getElementById("edit-rotate-input");

  const syncHeight = (v) => {
    const h = Math.max(3, Number(v) || 3);
    hRange.value = String(h);
    hInput.value = String(h);
    handlers.onHeightChange(h);
  };
  const syncBase = (v) => {
    const b = Math.max(0, Number(v) || 0);
    bRange.value = String(b);
    bInput.value = String(b);
    handlers.onBaseChange(b);
  };

  hRange.addEventListener("input", () => syncHeight(hRange.value));
  hInput.addEventListener("change", () => syncHeight(hInput.value));
  bRange.addEventListener("input", () => syncBase(bRange.value));
  bInput.addEventListener("change", () => syncBase(bInput.value));
  fnSelect.addEventListener("change", () => handlers.onFunctionChange(fnSelect.value));

  document.getElementById("btn-rotate-left").addEventListener("click", () => handlers.onRotate(-Number(rotateInput.value || 5)));
  document.getElementById("btn-rotate-right").addEventListener("click", () => handlers.onRotate(Number(rotateInput.value || 5)));
  document.getElementById("btn-export-scenario").addEventListener("click", handlers.onExport);
  document.getElementById("btn-save-scenario").addEventListener("click", handlers.onSaveLocal);
  document.getElementById("btn-reset-scenario").addEventListener("click", () => handlers.onResetScenario && handlers.onResetScenario());
  document.getElementById("btn-delete-selected").addEventListener("click", handlers.onDeleteSelected);
  document.getElementById("btn-mode-select").addEventListener("click", () => handlers.onModeChange("SELECT"));
  document.getElementById("btn-mode-add").addEventListener("click", () => handlers.onModeChange("ADD"));
  document.getElementById("btn-mode-vertex").addEventListener("click", () => handlers.onModeChange("VERTEX"));
  document.getElementById("btn-generate-cluster").addEventListener("click", () => handlers.onGenerateCluster && handlers.onGenerateCluster());
  document.getElementById("btn-ai-allocate").addEventListener("click", () => {
    const prompt = document.getElementById("ai-vision-input").value || "";
    if (handlers.onAiAllocate) handlers.onAiAllocate(prompt);
  });

  const audienceSelect = document.getElementById("ai-audience-select");
  const audienceOther = document.getElementById("ai-audience-other");
  if (audienceSelect && audienceOther) {
    const syncAudienceOther = () => {
      audienceOther.style.display = audienceSelect.value === "其它" ? "block" : "none";
    };
    audienceSelect.addEventListener("change", syncAudienceOther);
    syncAudienceOther();
  }

  const economicsUI = createEconomicsUI(handlers, numFrom);

  return {
    setEconomicsInputs: economicsUI.setEconomicsInputs,
    pushEcoParams: economicsUI.pushEcoParams,
    renderDashboard: economicsUI.renderDashboard,
    renderSelection(selection) {
      document.getElementById("edit-selected-id").textContent = selection ? selection.scenarioId : "-";
      if (!selection || selection.multi) return;
      hRange.value = String(Math.round(selection.height));
      hInput.value = String(Math.round(selection.height));
      bRange.value = String(Math.round(selection.base));
      bInput.value = String(Math.round(selection.base));
      fnSelect.value = selection.functionType;
    },
    renderMode(mode) {
      ["select", "add", "vertex"].forEach((id) => {
        const btn = document.getElementById(`btn-mode-${id}`);
        btn.style.outline = mode === id.toUpperCase() ? "2px solid #7cc4ff" : "none";
      });
      document.getElementById("edit-mode").textContent = mode;
    },
    renderAiSummary(text) {
      const el = document.getElementById("ai-allocation-summary");
      if (el) el.textContent = text || "";
    },
    renderAiDetails(text) {
      const el = document.getElementById("ai-allocation-details");
      if (el) el.textContent = text || "";
    },
  };
}
