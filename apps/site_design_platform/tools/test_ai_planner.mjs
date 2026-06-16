import { runFunctionalPlanning } from "../frontend/src/ai-function-planner.js";
import { createScenarioStore } from "../frontend/src/scenario-model.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function polygon(x0, y0, x1, y1) {
  return {
    type: "Polygon",
    coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
  };
}

function testAiPlanner() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "cbd_1", Height: 60, functionType: "OFFICE" }, geometry: polygon(121.0, 31.0, 121.0002, 31.0002) },
      { type: "Feature", properties: { _scenarioId: "res_1", Height: 24, functionType: "OFFICE" }, geometry: polygon(121.002, 31.0, 121.0022, 31.0002) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(120.999, 30.999, 121.001, 31.001) },
      { type: "Feature", properties: { layer: "Z_RES" }, geometry: polygon(121.0015, 30.999, 121.003, 31.001) },
    ],
  };
  const out = runFunctionalPlanning("兼具商务效率与安静居住", blocks, parcels);
  const byId = Object.fromEntries(out.assignments.map((x) => [x.id, x]));
  assert(byId.cbd_1 && byId.cbd_1.functionType === "OFFICE", "CBD block should prefer OFFICE");
  assert(byId.res_1 && byId.res_1.functionType === "RESIDENTIAL", "Residential block should prefer RESIDENTIAL");
  return out.summary;
}

function testConflictRules() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "res_high_1", Height: 78, functionType: "OFFICE" }, geometry: polygon(121.101, 31.001, 121.1012, 31.0012) },
      { type: "Feature", properties: { _scenarioId: "res_mid_1", Height: 30, functionType: "OFFICE" }, geometry: polygon(121.1013, 31.001, 121.1015, 31.0012) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { layer: "Z_RES" }, geometry: polygon(121.1005, 31.0005, 121.102, 31.002) },
    ],
  };
  const out = runFunctionalPlanning("提升夜间活力并保持居住安静", blocks, parcels);
  const byId = Object.fromEntries(out.assignments.map((x) => [x.id, x]));
  const hi = byId.res_high_1?.functionType || "";
  const mid = byId.res_mid_1?.functionType || "";
  assert(hi !== "LEISURE_COMMERCIAL" && hi !== "CENTER_COMMERCIAL", "Residential high band must avoid commercial leisure functions");
  assert(mid !== "LEISURE_COMMERCIAL", "Residential zone should avoid leisure-commercial as dominant choice");
  return `${hi}/${mid}`;
}

function testLeisureGroundBias() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "tod_low_1", Height: 12, functionType: "OFFICE" }, geometry: polygon(121.201, 31.001, 121.2012, 31.0012) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { layer: "Z_TOD" }, geometry: polygon(121.2005, 31.0005, 121.202, 31.002) },
    ],
  };
  const out = runFunctionalPlanning("强调消费活力和公共开放", blocks, parcels);
  const fn = out.assignments[0]?.functionType || "";
  assert(fn !== "OFFICE", "Leisure/TOD low band should not default to OFFICE");
  return fn;
}

function testGroundEconomics() {
  const fc = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { id: "g1", Height: 3, Base: 0, functionType: "GROUND" }, geometry: polygon(121.01, 31.01, 121.0103, 31.0103) },
      { type: "Feature", properties: { id: "gr1", Height: 3, Base: 0, functionType: "GREEN" }, geometry: polygon(121.02, 31.01, 121.0202, 31.0102) },
    ],
  };
  const store = createScenarioStore(fc);
  const stats = store.getStats({});
  assert(Number(stats.eco.ground_area_sqm) > 0, "GROUND area should be counted");
  assert(Number(stats.by_function_financial?.GROUND?.cost || 0) > 0, "GROUND cost should be counted");
  return `ground_area=${Math.round(stats.eco.ground_area_sqm)} sqm, ground_cost=${Math.round(stats.by_function_financial.GROUND.cost)}`;
}

function testEmptyPromptFallback() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "u1", Height: 33, functionType: "OFFICE" }, geometry: polygon(122.01, 31.01, 122.0102, 31.0102) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(122.0, 31.0, 122.02, 31.02) }],
  };
  const out = runFunctionalPlanning("", blocks, parcels);
  const fn = out.assignments[0]?.functionType || "";
  assert(fn.length > 0, "Empty prompt should still produce an assignment");
  return fn;
}

function testUnknownZoneFallback() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "x1", Height: 25, functionType: "OFFICE" }, geometry: polygon(123.01, 32.01, 123.0102, 32.0102) },
    ],
  };
  const parcels = { type: "FeatureCollection", features: [] };
  const out = runFunctionalPlanning("打造均衡复合社区", blocks, parcels);
  assert(out.assignments[0]?.zone === "UNKNOWN", "No parcel should map to UNKNOWN zone");
  assert(Boolean(out.assignments[0]?.functionType), "UNKNOWN zone must still assign function");
  return `${out.assignments[0].zone}:${out.assignments[0].functionType}`;
}

function testNightVsResidentialGuardrail() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "rnight1", Height: 50, functionType: "OFFICE" }, geometry: polygon(124.01, 31.01, 124.0102, 31.0102) },
      { type: "Feature", properties: { _scenarioId: "rnight2", Height: 12, functionType: "OFFICE" }, geometry: polygon(124.0103, 31.01, 124.0105, 31.0102) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { layer: "Z_RES" }, geometry: polygon(124.0, 31.0, 124.02, 31.02) }],
  };
  const out = runFunctionalPlanning("夜间活力越强越好", blocks, parcels);
  const fns = out.assignments.map((x) => x.functionType);
  assert(!fns.includes("LEISURE_COMMERCIAL"), "Residential guardrail should block leisure commercial even for night-heavy prompts");
  return fns.join(",");
}

function testDeterminism() {
  const blocks = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { _scenarioId: "d1", Height: 66, functionType: "OFFICE" }, geometry: polygon(125.01, 31.01, 125.0102, 31.0102) },
      { type: "Feature", properties: { _scenarioId: "d2", Height: 18, functionType: "OFFICE" }, geometry: polygon(125.011, 31.011, 125.0112, 31.0112) },
    ],
  };
  const parcels = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(125.0, 31.0, 125.012, 31.012) },
    ],
  };
  const p = "打造高效率商务门户并兼顾公共开放";
  const a = runFunctionalPlanning(p, blocks, parcels).assignments.map((x) => x.functionType).join("|");
  const b = runFunctionalPlanning(p, blocks, parcels).assignments.map((x) => x.functionType).join("|");
  assert(a === b, "Same inputs should produce deterministic output");
  return a;
}

function testGroundFunctionDefaults() {
  const fc = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { id: "gg1", Height: 3, Base: 0, functionType: "GROUND" }, geometry: polygon(126.01, 31.01, 126.0102, 31.0102) },
    ],
  };
  const store = createScenarioStore(fc);
  const s = store.toScenarioJSON();
  const b = s.blocks[0];
  assert(b.function === "GROUND", "GROUND function should persist");
  assert(Number(b.cost_params?.hard_cost_per_sqm) > 0, "GROUND should carry default cost params");
  return `${b.function}:${b.cost_params.hard_cost_per_sqm}`;
}

try {
  const s1 = testAiPlanner();
  const s3 = testConflictRules();
  const s4 = testLeisureGroundBias();
  const s2 = testGroundEconomics();
  const s5 = testEmptyPromptFallback();
  const s6 = testUnknownZoneFallback();
  const s7 = testNightVsResidentialGuardrail();
  const s8 = testDeterminism();
  const s9 = testGroundFunctionDefaults();
  console.log("PASS testAiPlanner:", s1);
  console.log("PASS testConflictRules:", s3);
  console.log("PASS testLeisureGroundBias:", s4);
  console.log("PASS testGroundEconomics:", s2);
  console.log("PASS testEmptyPromptFallback:", s5);
  console.log("PASS testUnknownZoneFallback:", s6);
  console.log("PASS testNightVsResidentialGuardrail:", s7);
  console.log("PASS testDeterminism:", s8);
  console.log("PASS testGroundFunctionDefaults:", s9);
} catch (err) {
  console.error("FAIL:", err.message);
  process.exit(1);
}
