import fs from "node:fs";
import path from "node:path";
import { runFloorStackPlanning } from "../frontend/src/ai-floor-planner.js";
import { createScenarioStore } from "../frontend/src/scenario-model.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  const p = path.resolve(process.cwd(), rel);
  return fs.readFileSync(p, "utf-8");
}

function polygon(x0, y0, x1, y1) {
  return {
    type: "Polygon",
    coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
  };
}

function testUiContracts() {
  const html = read("apps/site_design_platform/frontend/index.html");
  const ui = read("apps/site_design_platform/frontend/src/ui-controller.js");
  const aiUi = read("apps/site_design_platform/frontend/src/ai-ui.js");
  const ecoUi = read("apps/site_design_platform/frontend/src/economics-ui.js");
  const main = read("apps/site_design_platform/frontend/src/main.js");
  const lm = read("apps/site_design_platform/frontend/src/layer-manager.js");

  [
    "btn-ai-allocate",
    "ai-vision-input",
    "ai-allocation-summary",
    "ai-allocation-details",
    "eco-fn-ground-sale",
    "eco-fn-ground-rent",
    "eco-fn-ground-hard",
    "eco-fn-ground-soft",
  ].forEach((id) => assert(html.includes(`id=\"${id}\"`), `missing html id: ${id}`));

  assert(aiUi.includes("onAiAllocate"), "ai-ui missing onAiAllocate binding");
  assert(ui.includes("renderAiSummary"), "ui-controller missing renderAiSummary");
  assert(ui.includes("renderAiDetails"), "ui-controller missing renderAiDetails");
  assert(ecoUi.includes("functionOverrides") && ecoUi.includes("GROUND"), "economics-ui missing GROUND economics override");

  assert(main.includes("requestFloorPlanWithAgent"), "main.js missing AI agent client import/use");
  assert(main.includes("createZoneInsightOverlay"), "main.js missing zone insight overlay wiring");
  assert(main.includes("onAiAllocate"), "main.js missing AI allocate handler");

  assert(lm.includes("rhino-ground-fill"), "layer-manager missing rhino-ground layer");
  assert(lm.includes("#c6ccd3") || lm.includes("#9aa3ad"), "GROUND color not set to gray palette");
  return "ui contracts ok";
}

function testAiApplyFlowSimulation() {
  const blocksFc = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { id: "blk_1", Height: 54, Base: 0, functionType: "RESIDENTIAL" }, geometry: polygon(121.5, 31.2, 121.5002, 31.2002) },
      { type: "Feature", properties: { id: "blk_2", Height: 24, Base: 0, functionType: "RESIDENTIAL" }, geometry: polygon(121.502, 31.2, 121.5022, 31.2002) },
      { type: "Feature", properties: { id: "blk_3", Height: 18, Base: 0, functionType: "OFFICE" }, geometry: polygon(121.504, 31.2, 121.5042, 31.2002) },
    ],
  };
  const parcelsFc = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(121.499, 31.199, 121.501, 31.201) },
      { type: "Feature", properties: { layer: "Z_RES" }, geometry: polygon(121.5015, 31.199, 121.505, 31.201) },
    ],
  };

  const store = createScenarioStore(JSON.parse(JSON.stringify(blocksFc)));
  const plan = runFloorStackPlanning("强化商务效率并保持居住安静", store.getFeatureCollection(), parcelsFc);
  let mappedCount = 0;
  const dominantSet = new Set();
  plan.outputs.forEach((a) => {
    const item = store.getById(a.id);
    if (!item) return;
    const dominant = String(a.dominant || "");
    if (!dominant) return;
    mappedCount += 1;
    dominantSet.add(dominant);
  });
  assert(mappedCount === 3, "All blocks should get dominant floor-stack suggestion");
  assert(dominantSet.size >= 1, "Dominant suggestions should be present");
  return `dominant_mapped=${mappedCount}, dominant=${Array.from(dominantSet).join("|")}`;
}

function testRhinoReplayContract() {
  const p = path.resolve(process.cwd(), "data/site_design_platform/scenarios/rhino_live.json");
  if (!fs.existsSync(p)) return "rhino_live.json not found (skipped)";
  const payload = JSON.parse(fs.readFileSync(p, "utf-8"));
  assert(Array.isArray(payload.blocks), "rhino_live.json blocks[] missing");
  const blockCount = payload.blocks.length;
  const rg = payload.rhino_ground;
  const groundCount = rg && Array.isArray(rg.features) ? rg.features.length : 0;
  return `rhino replay ok: blocks=${blockCount}, rhino_ground=${groundCount}`;
}

try {
  const a = testUiContracts();
  const b = testAiApplyFlowSimulation();
  const c = testRhinoReplayContract();
  console.log("PASS testUiContracts:", a);
  console.log("PASS testAiApplyFlowSimulation:", b);
  console.log("PASS testRhinoReplayContract:", c);
} catch (err) {
  console.error("FAIL:", err.message);
  process.exit(1);
}
