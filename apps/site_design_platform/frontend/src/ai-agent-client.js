import { runFloorStackPlanning } from "./ai-floor-planner.js";

export async function requestFloorPlanWithAgent({ prompt, blocksFc, parcelsFc, audienceProfile }) {
  const endpoint = window.__SITE_AI_ENDPOINT__ || "/api/site-design/ai/floor-stack";
  const payload = { prompt, audience_profile: audienceProfile || "", blocks: blocksFc, parcels: parcelsFc };

  try {
    const r = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`AI agent failed: ${r.status} ${t}`);
    }
    const out = await r.json();
    if (!out || !Array.isArray(out.outputs)) throw new Error("AI agent response missing outputs[]");
    return { ...out, _engine: "agent" };
  } catch (err) {
    const local = runFloorStackPlanning(prompt, blocksFc, parcelsFc);
    return { ...local, _engine: "fallback", _fallbackReason: String(err?.message || err || "unknown"), _agentOk: false };
  }
}

function fallbackAudienceTemplate(prompt) {
  const t = String(prompt || "");
  const tags = [];
  if (/(商务|办公|金融|总部|效率)/.test(t)) tags.push("商务办公人群");
  if (/(年轻|潮流|夜间|消费|餐饮)/.test(t)) tags.push("年轻消费人群");
  if (/(创意|文创|艺术|设计|展览)/.test(t)) tags.push("创意产业人群");
  if (/(居住|宜居|家庭|社区|安静)/.test(t)) tags.push("家庭居住人群");
  if (/(旅游|地标|游客|门户)/.test(t)) tags.push("游客与外来访客");
  if (!tags.length) tags.push("商务办公人群", "家庭居住人群");
  return {
    audiences: tags.slice(0, 3),
    source: "fallback_template",
    note: "agent failed, fallback template applied",
  };
}

export async function requestAudienceCompletion(prompt) {
  const endpoint = "/api/site-design/ai/complete-audience";
  try {
    const r = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) throw new Error(await r.text());
    const out = await r.json();
    if (!out || !Array.isArray(out.audiences) || !out.audiences.length) return fallbackAudienceTemplate(prompt);
    return out;
  } catch {
    return fallbackAudienceTemplate(prompt);
  }
}
