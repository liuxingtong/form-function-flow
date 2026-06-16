const KEYWORDS = {
  business: ["商务", "总部", "办公", "金融", "企业", "效率"],
  leisure: ["休闲", "消费", "商业", "街区", "餐饮", "游逛"],
  quiet: ["安静", "宜居", "居住", "社区", "家庭", "宁静"],
  creative: ["文创", "创意", "艺术", "设计", "展览", "文化"],
  night: ["夜间", "夜生活", "夜晚", "周末", "演艺", "剧场"],
  open: ["开放", "公共", "广场", "滨水", "慢行", "步行"],
  landmark: ["地标", "门户", "形象", "城市中心"],
};

const PROGRAM = {
  CBD: {
    ground: ["高端零售", "商务餐饮", "酒店大堂", "办公大堂", "商务展示"],
    podium: ["商务会展", "高端零售", "商务餐饮"],
    mid: ["标准办公", "总部办公", "商务酒店"],
    high: ["总部办公", "商务酒店", "标准办公"],
    roof: ["商务会所", "观景餐厅", "公共露台"],
  },
  LEISURE: {
    ground: ["生活方式零售", "特色餐饮", "咖啡", "精品酒店大堂"],
    podium: ["休闲娱乐", "剧场演艺", "生活方式零售"],
    mid: ["精品酒店", "服务式公寓", "创意办公"],
    high: ["精品酒店", "服务式公寓", "创意办公"],
    roof: ["露台餐饮", "公共活动", "屋顶花园"],
  },
  RESIDENTIAL: {
    ground: ["社区零售", "社区服务", "住宅大堂", "邻里空间"],
    podium: ["社区服务", "文化教育", "少量社区商业"],
    mid: ["高端住宅", "人才公寓", "社区服务"],
    high: ["高端住宅", "人才公寓", "社区服务"],
    roof: ["社区花园", "安静共享露台", "设备空间"],
  },
  CREATIVE: {
    ground: ["文化展览", "咖啡", "开放工作室", "创意办公入口"],
    podium: ["文化展览", "发布空间", "设计工作室", "文化教育"],
    mid: ["创意办公", "人才公寓", "精品酒店"],
    high: ["创意办公", "人才公寓", "精品酒店"],
    roof: ["展演平台", "交流活动露台", "屋顶花园"],
  },
  UNKNOWN: {
    ground: ["社区服务", "公共展示", "生活方式零售"],
    podium: ["公共服务", "文化展览", "标准办公"],
    mid: ["标准办公", "服务式公寓", "高端住宅"],
    high: ["标准办公", "服务式公寓", "高端住宅"],
    roof: ["公共露台", "屋顶花园", "设备空间"],
  },
};

const SEGMENT_WEIGHT = { ground: 0.22, podium: 0.24, mid: 0.28, high: 0.2, roof: 0.06 };

function ring(feature) {
  return feature?.geometry?.type === "Polygon" ? (feature.geometry.coordinates?.[0] || []) : [];
}

function centroid(feature) {
  const r = ring(feature);
  if (!r.length) return null;
  let x = 0; let y = 0; let n = 0;
  r.slice(0, -1).forEach((p) => { x += p[0]; y += p[1]; n += 1; });
  return n ? [x / n, y / n] : r[0];
}

function pointInRing(lng, lat, r) {
  let inside = false;
  let j = r.length - 1;
  for (let i = 0; i < r.length; i += 1) {
    const xi = r[i][0]; const yi = r[i][1];
    const xj = r[j][0]; const yj = r[j][1];
    const hit = ((yi > lat) !== (yj > lat)) && (lng < ((xj - xi) * (lat - yi)) / ((yj - yi) || 1e-12) + xi);
    if (hit) inside = !inside;
    j = i;
  }
  return inside;
}

function zoneOf(block, parcelsFc) {
  const c = centroid(block);
  if (!c) return "UNKNOWN";
  let zone = "UNKNOWN";
  (parcelsFc?.features || []).forEach((p) => {
    const layer = String(p?.properties?.layer || p?.properties?.zone_id || "").toUpperCase();
    if (!layer.startsWith("Z_")) return;
    const r = ring(p);
    if (!r.length || !pointInRing(c[0], c[1], r)) return;
    if (layer === "Z_CBD") zone = "CBD";
    else if (layer === "Z_TOD") zone = "LEISURE";
    else if (layer === "Z_OFC") zone = "CREATIVE";
    else if (layer === "Z_RES") zone = "RESIDENTIAL";
  });
  return zone;
}

function extractIntent(prompt) {
  const t = String(prompt || "");
  const out = { business: 50, leisure: 50, quiet: 50, creative: 50, night: 50, open: 50, landmark: 50 };
  Object.entries(KEYWORDS).forEach(([k, arr]) => {
    let hits = 0;
    arr.forEach((w) => { if (t.includes(w)) hits += 1; });
    if (hits > 0) out[k] = Math.min(100, 50 + hits * 14);
  });
  return out;
}

function segmentsForHeight(h) {
  const seg = ["ground"];
  if (h >= 18) seg.push("podium");
  if (h >= 36) seg.push("mid");
  if (h >= 60) seg.push("high");
  seg.push("roof");
  return seg;
}

function hardConflict(zone, segment, fn) {
  if (zone === "RESIDENTIAL" && (segment === "mid" || segment === "high")) {
    if (["生活方式零售", "高端零售", "特色餐饮", "休闲娱乐", "剧场演艺", "商务会展", "商务餐饮"].includes(fn)) return true;
  }
  if (zone === "CBD" && segment === "high" && ["社区服务", "社区零售"].includes(fn)) return true;
  if (zone === "LEISURE" && segment === "ground" && fn === "标准办公") return true;
  if (zone === "CREATIVE" && segment === "ground" && fn === "高端住宅") return true;
  return false;
}

function scoreFn(fn, zone, segment, intent) {
  let s = 55;
  if (["总部办公", "标准办公", "商务酒店", "商务会展", "办公大堂"].includes(fn)) s += intent.business * 0.22;
  if (["特色餐饮", "生活方式零售", "休闲娱乐", "剧场演艺", "露台餐饮"].includes(fn)) s += intent.leisure * 0.2 + intent.night * 0.15;
  if (["高端住宅", "人才公寓", "社区服务", "住宅大堂", "邻里空间", "社区花园", "安静共享露台"].includes(fn)) s += intent.quiet * 0.22;
  if (["文化展览", "设计工作室", "开放工作室", "创意办公", "发布空间", "展演平台"].includes(fn)) s += intent.creative * 0.22;
  if (["公共露台", "公共活动", "公共展示", "咖啡", "商务展示"].includes(fn)) s += intent.open * 0.18;
  if (["总部办公", "商务酒店", "观景餐厅"].includes(fn)) s += intent.landmark * 0.14;

  if (zone === "CBD" && ["总部办公", "标准办公"].includes(fn) && (segment === "mid" || segment === "high")) s += 14;
  if (zone === "LEISURE" && ["特色餐饮", "生活方式零售", "休闲娱乐"].includes(fn) && (segment === "ground" || segment === "podium")) s += 14;
  if (zone === "RESIDENTIAL" && ["高端住宅", "人才公寓", "社区服务"].includes(fn)) s += 16;
  if (zone === "CREATIVE" && ["文化展览", "设计工作室", "创意办公"].includes(fn)) s += 14;

  if (hardConflict(zone, segment, fn)) s -= 120;
  return s;
}

function chooseTwo(candidates, zone, segment, intent) {
  const scored = candidates
    .map((fn) => ({ fn, score: scoreFn(fn, zone, segment, intent) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score);
  const primary = scored[0]?.fn || candidates[0];
  const secondary = scored.find((x) => x.fn !== primary)?.fn || null;
  return { primary, secondary, score: Math.round(scored[0]?.score || 50) };
}

function mapLabelToFunctionCode(label, zone) {
  if (["总部办公", "标准办公", "办公大堂", "创意办公", "创意办公入口", "设计工作室", "开放工作室", "商务展示"].includes(label)) return "OFFICE";
  if (["高端住宅", "人才公寓", "住宅大堂", "邻里空间"].includes(label)) return "RESIDENTIAL";
  if (["高端零售", "生活方式零售", "社区零售"].includes(label)) return "CENTER_COMMERCIAL";
  if (["特色餐饮", "休闲娱乐", "剧场演艺", "露台餐饮", "观景餐厅", "咖啡"].includes(label)) return "LEISURE_COMMERCIAL";
  if (["商务酒店", "精品酒店", "服务式公寓", "酒店大堂", "商务会所"].includes(label)) return zone === "CBD" ? "OFFICE" : "LEISURE_COMMERCIAL";
  if (["文化展览", "发布空间", "文化教育", "公共露台", "公共活动", "公共展示", "社区服务", "公共服务", "社区花园", "安静共享露台"].includes(label)) return "PUBLIC";
  if (["屋顶花园", "设备空间"].includes(label)) return "GREEN";
  return "PUBLIC";
}

function narrativeZoneName(zone, intent) {
  if (zone === "CBD") return intent.business >= 65 ? "商务引擎核" : "复合商务核";
  if (zone === "LEISURE") return intent.night >= 65 ? "夜间活力湾" : "公共消费带";
  if (zone === "RESIDENTIAL") return intent.quiet >= 65 ? "静享居住里" : "生活服务居住区";
  if (zone === "CREATIVE") return intent.creative >= 65 ? "创意展演谷" : "文创复合区";
  return "均衡复合区";
}

function zoneHeadline(zone, intent) {
  if (zone === "CBD") return intent.business >= 60 ? "该区以高效率商务和总部办公为主，首层强化商务展示与配套消费。" : "该区保持商务主导，并以公共界面提升活力。";
  if (zone === "LEISURE") return intent.night >= 60 ? "该区主打夜间与周末活力，首层和低层重点布置消费与演艺功能。" : "该区以休闲消费为主，形成全天候开放街区。";
  if (zone === "RESIDENTIAL") return intent.quiet >= 60 ? "该区坚持安静宜居，中高层以住宅为主，首层补充社区服务。" : "该区以居住稳定为底盘，适度配置生活服务与公共空间。";
  if (zone === "CREATIVE") return intent.creative >= 60 ? "该区以创意生产与文化展示协同，形成开放的文创界面。" : "该区兼顾创意办公与公共文化活动，保持复合弹性。";
  return "该区采用均衡复合策略，按楼层段逐步组织功能。";
}

function summarizeZoneMix(outputs, intent) {
  const zoneMix = {};
  outputs.forEach((o) => {
    if (!zoneMix[o.zone]) zoneMix[o.zone] = {};
    o.segments.forEach((s) => {
      const code = mapLabelToFunctionCode(s.primary, o.zone);
      const w = SEGMENT_WEIGHT[s.segment] || 0.2;
      zoneMix[o.zone][code] = (zoneMix[o.zone][code] || 0) + w;
    });
  });
  const zoneInsights = Object.entries(zoneMix).map(([zone, mix]) => {
    const total = Object.values(mix).reduce((a, b) => a + b, 0) || 1;
    const ratios = Object.entries(mix)
      .map(([k, v]) => ({ key: k, ratio: v / total }))
      .sort((a, b) => b.ratio - a.ratio);
    return {
      zone,
      ratios,
      headline: zoneHeadline(zone, intent),
      narrativeName: narrativeZoneName(zone, intent),
    };
  });
  return zoneInsights;
}

export function runFloorStackPlanning(prompt, blocksFc, parcelsFc) {
  const intent = extractIntent(prompt);
  const outputs = [];
  (blocksFc?.features || []).forEach((b, idx) => {
    const id = String(b?.properties?._scenarioId || b?.properties?.id || `blk_${idx + 1}`);
    const h = Number(b?.properties?.Height ?? b?.properties?.height ?? 24);
    const zone = zoneOf(b, parcelsFc);
    const segments = segmentsForHeight(h).map((seg) => {
      const candidates = PROGRAM[zone]?.[seg] || PROGRAM.UNKNOWN[seg];
      const picked = chooseTwo(candidates, zone, seg, intent);
      const reason = `${zone}分区 + ${seg}层段 + 愿景权重匹配`;
      const primaryCode = mapLabelToFunctionCode(picked.primary, zone);
      const secondaryCode = picked.secondary ? mapLabelToFunctionCode(picked.secondary, zone) : null;
      return {
        segment: seg,
        primary: picked.primary,
        secondary: picked.secondary,
        primaryCode,
        secondaryCode,
        score: picked.score,
        reason,
      };
    });
    const dominant = segments.find((x) => x.segment === "mid" || x.segment === "high" || x.segment === "podium")?.primary || segments[0]?.primary;
    const dominantCode = mapLabelToFunctionCode(dominant, zone);
    outputs.push({ id, zone, height: h, segments, dominant, dominantCode });
  });
  const zoneInsights = summarizeZoneMix(outputs, intent);
  return {
    intent,
    outputs,
    zoneInsights,
    summary: `分层排布完成：${outputs.length}个体块。权重 商务${intent.business}/休闲${intent.leisure}/居住安静${intent.quiet}/文创${intent.creative}/夜间${intent.night}/开放${intent.open}`,
  };
}
