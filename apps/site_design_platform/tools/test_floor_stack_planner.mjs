import { runFloorStackPlanning } from "../frontend/src/ai-floor-planner.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function polygon(x0, y0, x1, y1) {
  return { type: "Polygon", coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] };
}

function testSegmentsByHeight() {
  const blocks = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { _scenarioId: "h1", Height: 12 }, geometry: polygon(121, 31, 121.0001, 31.0001) },
    { type: "Feature", properties: { _scenarioId: "h2", Height: 72 }, geometry: polygon(121.001, 31, 121.0011, 31.0001) },
  ] };
  const parcels = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(120.99, 30.99, 121.01, 31.01) },
  ] };
  const out = runFloorStackPlanning("商务门户", blocks, parcels);
  const b1 = out.outputs.find((x) => x.id === "h1");
  const b2 = out.outputs.find((x) => x.id === "h2");
  assert(b1.segments.length === 2, "12m block should have ground+roof");
  assert(b2.segments.length >= 5, "72m block should have full bands");
  return `${b1.segments.map((x) => x.segment).join(",")} | ${b2.segments.map((x) => x.segment).join(",")}`;
}

function testResidentialMidHighGuardrail() {
  const blocks = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { _scenarioId: "r1", Height: 66 }, geometry: polygon(122, 31, 122.0001, 31.0001) },
  ] };
  const parcels = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { layer: "Z_RES" }, geometry: polygon(121.99, 30.99, 122.01, 31.01) },
  ] };
  const out = runFloorStackPlanning("夜间活力越高越好", blocks, parcels);
  const segs = out.outputs[0].segments.filter((s) => s.segment === "mid" || s.segment === "high");
  const bad = ["生活方式零售", "高端零售", "特色餐饮", "休闲娱乐", "剧场演艺", "商务会展", "商务餐饮"];
  segs.forEach((s) => assert(!bad.includes(s.primary), `residential ${s.segment} should avoid noisy commercial primary`));
  return segs.map((s) => `${s.segment}:${s.primary}`).join(" / ");
}

function testOutputExplainability() {
  const blocks = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { _scenarioId: "c1", Height: 48 }, geometry: polygon(123, 31, 123.0001, 31.0001) },
  ] };
  const parcels = { type: "FeatureCollection", features: [
    { type: "Feature", properties: { layer: "Z_CBD" }, geometry: polygon(122.99, 30.99, 123.01, 31.01) },
  ] };
  const out = runFloorStackPlanning("商务效率与公共开放兼顾", blocks, parcels);
  const seg = out.outputs[0].segments[0];
  assert(typeof seg.reason === "string" && seg.reason.length > 0, "segment should include reason");
  assert(Number.isFinite(seg.score), "segment should include numeric score");
  return `${seg.segment}:${seg.primary}:${seg.score}`;
}

try {
  console.log("PASS testSegmentsByHeight:", testSegmentsByHeight());
  console.log("PASS testResidentialMidHighGuardrail:", testResidentialMidHighGuardrail());
  console.log("PASS testOutputExplainability:", testOutputExplainability());
} catch (err) {
  console.error("FAIL:", err.message);
  process.exit(1);
}
