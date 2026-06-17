import { FUNCTION_COLORS } from "./config.js";

const FUNCTION_FIELDS = {
  CENTER_OFFICE: { prefix: "eco-fn-center-office", label: "CENTER_OFFICE" },
  SMALL_OFFICE: { prefix: "eco-fn-small-office", label: "SMALL_OFFICE" },
  APARTMENT: { prefix: "eco-fn-apartment", label: "APARTMENT" },
  HOTEL: { prefix: "eco-fn-hotel", label: "HOTEL" },
  RESIDENTIAL: { prefix: "eco-fn-res", label: "RESIDENTIAL" },
  CENTER_COMMERCIAL: { prefix: "eco-fn-cc", label: "CENTER_COMMERCIAL" },
  LEISURE_COMMERCIAL: { prefix: "eco-fn-lc", label: "LEISURE_COMMERCIAL" },
  PUBLIC: { prefix: "eco-fn-public", label: "PUBLIC" },
  GROUND: { prefix: "eco-fn-ground", label: "GROUND" },
};

const PROJECT_INPUT_IDS = [
  "eco-input-years",
  "eco-input-build-years",
  "eco-input-land-cost-per-gfa",
  "eco-input-finance-rate",
  "eco-input-debt-ratio",
  "eco-input-tax-ratio",
  "eco-input-sale-marketing-ratio",
  "eco-input-lease-marketing-ratio",
  "eco-input-target-profit-ratio",
  "eco-input-public-value-per-sqm",
  "eco-input-carbon-factor",
  "eco-input-runoff-factor",
  "eco-input-carbon-price",
  "eco-input-runoff-price",
];

const FUNCTION_SUFFIXES = ["sale", "rent", "hard", "soft", "infra", "cont", "occ", "opex", "cap"];

function readFunctionOverrides(numFrom) {
  return Object.fromEntries(Object.entries(FUNCTION_FIELDS).map(([fn, cfg]) => ([fn, {
    sale_price_per_sqm: numFrom(`${cfg.prefix}-sale`),
    rent_price_per_sqm_year: numFrom(`${cfg.prefix}-rent`),
    hard_cost_per_sqm: numFrom(`${cfg.prefix}-hard`),
    soft_cost_ratio: numFrom(`${cfg.prefix}-soft`),
    infra_cost_per_sqm: numFrom(`${cfg.prefix}-infra`),
    contingency_ratio: numFrom(`${cfg.prefix}-cont`),
    occupancy: numFrom(`${cfg.prefix}-occ`),
    opex_ratio: numFrom(`${cfg.prefix}-opex`),
    cap_rate: numFrom(`${cfg.prefix}-cap`),
  }])));
}

function setIf(id, value) {
  if (value === undefined || value === null || !Number.isFinite(Number(value))) return;
  const el = document.getElementById(id);
  if (el) el.value = String(value);
}

function formatMoneyYi(value) {
  const n = Number(value || 0);
  if (Math.abs(n) >= 1e4) return `${(n / 1e8).toFixed(2)} 亿`;
  return `${n.toFixed(0)} 元`;
}

function formatGfaWan(value) {
  const n = Number(value || 0);
  if (n >= 10000) return `${(n / 10000).toFixed(2)} 万㎡`;
  return `${Math.round(n).toLocaleString()} ㎡`;
}

function assessDataReasonability(stats) {
  const gfa = Number(stats.gfa || 0);
  const tdc = Number(stats.tdc || 0);
  const totalValue = Number(stats.total_value || 0);
  const profit = Number(stats.profit || 0);
  if (gfa <= 0 || tdc <= 0) return "";
  const costPerGfa = tdc / gfa;
  const valuePerGfa = totalValue / gfa;
  const poc = Number(stats.profit_on_cost || 0);
  const notes = [];
  if (costPerGfa < 35000) notes.push("单方成本偏低，请核对土地与建安参数");
  else if (costPerGfa > 90000) notes.push("单方成本偏高，可能高估了土地或建安");
  else notes.push("单方成本处于站城一体项目常见区间");
  if (valuePerGfa < 45000) notes.push("单方价值偏低");
  else if (valuePerGfa > 95000) notes.push("单方价值偏高，乐观情景下才较常见");
  else notes.push("单方价值与核心区混合开发量级相符");
  if (poc < 0) notes.push("利润为负，方案经济性不足");
  else if (poc > 0.35) notes.push(`利润率 ${(poc * 100).toFixed(1)}% 偏乐观，建议对照保守情景`);
  else notes.push(`利润率 ${(poc * 100).toFixed(1)}% 处于可接受区间`);
  return `数据校核：建面 ${formatGfaWan(gfa)} · 单方成本 ${(costPerGfa / 10000).toFixed(1)} 万/㎡ · 单方价值 ${(valuePerGfa / 10000).toFixed(1)} 万/㎡ · ${notes.join("；")}`;
}

const ECONOMIC_PRESETS = {
  conservative: {
    years: 5,
    buildYears: 2,
    landCostPerGfa: 30000,
    financeRate: 0.07,
    debtRatio: 0.55,
    taxRatio: 0.08,
    saleMarketingRatio: 0.04,
    leaseMarketingRatio: 0.03,
    targetProfitRatio: 0.15,
    publicValuePerSqm: 2200,
    ecoCarbonFactor: 0.0009,
    ecoRunoffFactor: 0.65,
    ecoCarbonPrice: 520,
    ecoRunoffPrice: 3.2,
    functionOverrides: {
      CENTER_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 5800, hard_cost_per_sqm: 23400, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.84, opex_ratio: 0.24, cap_rate: 0.058 },
      SMALL_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 4300, hard_cost_per_sqm: 21000, soft_cost_ratio: 0.19, infra_cost_per_sqm: 2520, contingency_ratio: 0.09, occupancy: 0.80, opex_ratio: 0.25, cap_rate: 0.060 },
      APARTMENT: { sale_price_per_sqm: 98000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 19500, soft_cost_ratio: 0.17, infra_cost_per_sqm: 2280, contingency_ratio: 0.08, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      HOTEL: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 5200, hard_cost_per_sqm: 25800, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3300, contingency_ratio: 0.10, occupancy: 0.76, opex_ratio: 0.34, cap_rate: 0.062 },
      RESIDENTIAL: { sale_price_per_sqm: 126000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 20400, soft_cost_ratio: 0.18, infra_cost_per_sqm: 2400, contingency_ratio: 0.09, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      CENTER_COMMERCIAL: { sale_price_per_sqm: 76000, rent_price_per_sqm_year: 6200, hard_cost_per_sqm: 26400, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3600, contingency_ratio: 0.10, occupancy: 0.80, opex_ratio: 0.28, cap_rate: 0.058 },
      LEISURE_COMMERCIAL: { sale_price_per_sqm: 46000, rent_price_per_sqm_year: 3800, hard_cost_per_sqm: 24600, soft_cost_ratio: 0.20, infra_cost_per_sqm: 3000, contingency_ratio: 0.10, occupancy: 0.74, opex_ratio: 0.32, cap_rate: 0.064 },
      PUBLIC: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 900, hard_cost_per_sqm: 23000, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.60, opex_ratio: 0.28, cap_rate: 0.060 },
      GROUND: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 3600, soft_cost_ratio: 0.10, infra_cost_per_sqm: 2700, contingency_ratio: 0.06, occupancy: 0.00, opex_ratio: 0.00, cap_rate: 0.060 },
    },
  },
  base: {
    years: 5,
    buildYears: 2,
    landCostPerGfa: 18000,
    financeRate: 0.055,
    debtRatio: 0.55,
    taxRatio: 0.06,
    saleMarketingRatio: 0.03,
    leaseMarketingRatio: 0.02,
    targetProfitRatio: 0.15,
    publicValuePerSqm: 2200,
    ecoCarbonFactor: 0.0009,
    ecoRunoffFactor: 0.65,
    ecoCarbonPrice: 520,
    ecoRunoffPrice: 3.2,
    functionOverrides: {
      CENTER_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 7200, hard_cost_per_sqm: 23400, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.91, opex_ratio: 0.24, cap_rate: 0.050 },
      SMALL_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 5550, hard_cost_per_sqm: 21000, soft_cost_ratio: 0.19, infra_cost_per_sqm: 2520, contingency_ratio: 0.09, occupancy: 0.88, opex_ratio: 0.25, cap_rate: 0.052 },
      APARTMENT: { sale_price_per_sqm: 126000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 19500, soft_cost_ratio: 0.17, infra_cost_per_sqm: 2280, contingency_ratio: 0.08, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      HOTEL: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 6300, hard_cost_per_sqm: 25800, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3300, contingency_ratio: 0.10, occupancy: 0.86, opex_ratio: 0.34, cap_rate: 0.056 },
      RESIDENTIAL: { sale_price_per_sqm: 156000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 20400, soft_cost_ratio: 0.18, infra_cost_per_sqm: 2400, contingency_ratio: 0.09, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      CENTER_COMMERCIAL: { sale_price_per_sqm: 90000, rent_price_per_sqm_year: 7800, hard_cost_per_sqm: 26400, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3600, contingency_ratio: 0.10, occupancy: 0.88, opex_ratio: 0.28, cap_rate: 0.052 },
      LEISURE_COMMERCIAL: { sale_price_per_sqm: 54000, rent_price_per_sqm_year: 4800, hard_cost_per_sqm: 24600, soft_cost_ratio: 0.20, infra_cost_per_sqm: 3000, contingency_ratio: 0.10, occupancy: 0.82, opex_ratio: 0.32, cap_rate: 0.058 },
      PUBLIC: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 900, hard_cost_per_sqm: 21000, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.60, opex_ratio: 0.28, cap_rate: 0.060 },
      GROUND: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 3600, soft_cost_ratio: 0.10, infra_cost_per_sqm: 2700, contingency_ratio: 0.06, occupancy: 0.00, opex_ratio: 0.00, cap_rate: 0.060 },
    },
  },
  optimistic: {
    years: 5,
    buildYears: 2,
    landCostPerGfa: 14000,
    financeRate: 0.045,
    debtRatio: 0.55,
    taxRatio: 0.048,
    saleMarketingRatio: 0.027,
    leaseMarketingRatio: 0.017,
    targetProfitRatio: 0.15,
    publicValuePerSqm: 2400,
    ecoCarbonFactor: 0.0009,
    ecoRunoffFactor: 0.65,
    ecoCarbonPrice: 520,
    ecoRunoffPrice: 3.2,
    functionOverrides: {
      CENTER_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 9400, hard_cost_per_sqm: 23400, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.945, opex_ratio: 0.24, cap_rate: 0.043 },
      SMALL_OFFICE: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 7000, hard_cost_per_sqm: 21000, soft_cost_ratio: 0.19, infra_cost_per_sqm: 2520, contingency_ratio: 0.09, occupancy: 0.915, opex_ratio: 0.25, cap_rate: 0.046 },
      APARTMENT: { sale_price_per_sqm: 150000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 19500, soft_cost_ratio: 0.17, infra_cost_per_sqm: 2280, contingency_ratio: 0.08, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      HOTEL: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 7800, hard_cost_per_sqm: 25800, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3300, contingency_ratio: 0.10, occupancy: 0.90, opex_ratio: 0.34, cap_rate: 0.050 },
      RESIDENTIAL: { sale_price_per_sqm: 188000, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 20400, soft_cost_ratio: 0.18, infra_cost_per_sqm: 2400, contingency_ratio: 0.09, occupancy: 0.95, opex_ratio: 0.12, cap_rate: 0.055 },
      CENTER_COMMERCIAL: { sale_price_per_sqm: 106000, rent_price_per_sqm_year: 9800, hard_cost_per_sqm: 26400, soft_cost_ratio: 0.22, infra_cost_per_sqm: 3600, contingency_ratio: 0.10, occupancy: 0.915, opex_ratio: 0.28, cap_rate: 0.046 },
      LEISURE_COMMERCIAL: { sale_price_per_sqm: 68000, rent_price_per_sqm_year: 6200, hard_cost_per_sqm: 24600, soft_cost_ratio: 0.20, infra_cost_per_sqm: 3000, contingency_ratio: 0.10, occupancy: 0.86, opex_ratio: 0.32, cap_rate: 0.052 },
      PUBLIC: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 900, hard_cost_per_sqm: 18800, soft_cost_ratio: 0.20, infra_cost_per_sqm: 2700, contingency_ratio: 0.10, occupancy: 0.60, opex_ratio: 0.28, cap_rate: 0.060 },
      GROUND: { sale_price_per_sqm: 0, rent_price_per_sqm_year: 0, hard_cost_per_sqm: 3600, soft_cost_ratio: 0.10, infra_cost_per_sqm: 2700, contingency_ratio: 0.06, occupancy: 0.00, opex_ratio: 0.00, cap_rate: 0.060 },
    },
  },
};

export function createEconomicsUI(handlers, numFrom) {
  const toggle = document.getElementById("economics-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const body = document.getElementById("economics-body");
      if (!body) return;
      const open = body.style.display !== "none";
      body.style.display = open ? "none" : "block";
      toggle.textContent = open ? "展开" : "收起";
    });
  }

  const ecoPages = {
    overview: document.getElementById("eco-page-overview"),
    mix: document.getElementById("eco-page-mix"),
  };
  const ecoTabs = {
    overview: document.getElementById("eco-tab-overview"),
    mix: document.getElementById("eco-tab-mix"),
  };
  const setEcoTab = (name) => {
    Object.entries(ecoPages).forEach(([k, el]) => { if (el) el.style.display = k === name ? "block" : "none"; });
    Object.entries(ecoTabs).forEach(([k, btn]) => { if (btn) btn.style.outline = k === name ? "2px solid #7cc4ff" : "none"; });
  };
  Object.entries(ecoTabs).forEach(([k, btn]) => btn && btn.addEventListener("click", () => setEcoTab(k)));
  setEcoTab("overview");

  const collectEcoParams = () => ({
    years: numFrom("eco-input-years"),
    buildYears: numFrom("eco-input-build-years"),
    landCostPerGfa: numFrom("eco-input-land-cost-per-gfa"),
    financeRate: numFrom("eco-input-finance-rate"),
    debtRatio: numFrom("eco-input-debt-ratio"),
    taxRatio: numFrom("eco-input-tax-ratio"),
    saleMarketingRatio: numFrom("eco-input-sale-marketing-ratio"),
    leaseMarketingRatio: numFrom("eco-input-lease-marketing-ratio"),
    targetProfitRatio: numFrom("eco-input-target-profit-ratio"),
    publicValuePerSqm: numFrom("eco-input-public-value-per-sqm"),
    ecoCarbonFactor: numFrom("eco-input-carbon-factor"),
    ecoRunoffFactor: numFrom("eco-input-runoff-factor"),
    ecoCarbonPrice: numFrom("eco-input-carbon-price"),
    ecoRunoffPrice: numFrom("eco-input-runoff-price"),
    functionOverrides: readFunctionOverrides(numFrom),
  });

  const pushEcoParams = () => {
    const params = collectEcoParams();
    handlers.onEconomicsChange(params);
    return params;
  };

  const setEconomicsInputs = (params = {}) => {
    setIf("eco-input-years", params.years);
    setIf("eco-input-build-years", params.buildYears);
    setIf("eco-input-land-cost-per-gfa", params.landCostPerGfa);
    setIf("eco-input-finance-rate", params.financeRate);
    setIf("eco-input-debt-ratio", params.debtRatio);
    setIf("eco-input-tax-ratio", params.taxRatio);
    setIf("eco-input-sale-marketing-ratio", params.saleMarketingRatio);
    setIf("eco-input-lease-marketing-ratio", params.leaseMarketingRatio);
    setIf("eco-input-target-profit-ratio", params.targetProfitRatio);
    setIf("eco-input-public-value-per-sqm", params.publicValuePerSqm);
    setIf("eco-input-carbon-factor", params.ecoCarbonFactor);
    setIf("eco-input-runoff-factor", params.ecoRunoffFactor);
    setIf("eco-input-carbon-price", params.ecoCarbonPrice);
    setIf("eco-input-runoff-price", params.ecoRunoffPrice);

    const fo = params.functionOverrides || {};
    Object.entries(FUNCTION_FIELDS).forEach(([fn, cfg]) => {
      const v = fo[fn] || {};
      setIf(`${cfg.prefix}-sale`, v.sale_price_per_sqm);
      setIf(`${cfg.prefix}-rent`, v.rent_price_per_sqm_year);
      setIf(`${cfg.prefix}-hard`, v.hard_cost_per_sqm);
      setIf(`${cfg.prefix}-soft`, v.soft_cost_ratio);
      setIf(`${cfg.prefix}-infra`, v.infra_cost_per_sqm);
      setIf(`${cfg.prefix}-cont`, v.contingency_ratio);
      setIf(`${cfg.prefix}-occ`, v.occupancy);
      setIf(`${cfg.prefix}-opex`, v.opex_ratio);
      setIf(`${cfg.prefix}-cap`, v.cap_rate);
    });
  };

  [...PROJECT_INPUT_IDS, ...Object.values(FUNCTION_FIELDS).flatMap((cfg) => FUNCTION_SUFFIXES.map((suffix) => `${cfg.prefix}-${suffix}`))].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", pushEcoParams);
  });

  const applyPreset = (name) => {
    const preset = ECONOMIC_PRESETS[name];
    if (!preset) return;
    setEconomicsInputs(preset);
    pushEcoParams();
  };

  const presetConservative = document.getElementById("eco-preset-conservative");
  if (presetConservative) presetConservative.addEventListener("click", () => applyPreset("conservative"));
  const presetBase = document.getElementById("eco-preset-base");
  if (presetBase) presetBase.addEventListener("click", () => applyPreset("base"));
  const presetOptimistic = document.getElementById("eco-preset-optimistic");
  if (presetOptimistic) presetOptimistic.addEventListener("click", () => applyPreset("optimistic"));

  const recalcBtn = document.getElementById("eco-recalc");
  if (recalcBtn) {
    recalcBtn.addEventListener("click", () => {
      const params = pushEcoParams();
      if (handlers.onEconomicsRecalc) handlers.onEconomicsRecalc(params);
    });
  }

  return { setEconomicsInputs, pushEcoParams, renderDashboard };
}

/* ═══════════════════════════════════════════════════════════════
   CHART ENGINE — light glass · monochrome + gold accent
   ═══════════════════════════════════════════════════════════════ */

const _C = {
  gold: '#a07840', goldM: 'rgba(160,120,64,.18)',
  ink: '#1f2937', mid: '#64748b', dim: '#9ca3af',
  pos: '#374151', neg: '#dc2626',
  track: 'rgba(0,0,0,.07)', line: 'rgba(0,0,0,.10)',
};

function _fmtV(v) {
  const n = Number(v || 0);
  if (Math.abs(n) >= 1e4) return `${(n / 1e8).toFixed(1)}亿`;
  return n.toFixed(0);
}

function _tx(x, y, t, { fill = _C.ink, size = 10, a = 'middle', w = 'normal' } = {}) {
  return `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" font-family="'Segoe UI','Microsoft YaHei',sans-serif" text-anchor="${a}" font-weight="${w}">${t}</text>`;
}

/* ── Donut ring ──────────────────────────────────────────────── */
function _donut(segs, W = 320, H = 160) {
  const cx = 80, cy = H / 2, r = 60, ri = 40;
  const total = segs.reduce((s, x) => s + Math.abs(x.v), 0) || 1;
  let ang = -Math.PI / 2;
  const paths = segs.map((seg) => {
    const a0 = ang, a1 = ang + (Math.abs(seg.v) / total) * 2 * Math.PI;
    ang = a1;
    if (a1 - a0 < 0.001) return '';
    const lg = a1 - a0 > Math.PI ? 1 : 0;
    const [xoo, yoo] = [cx + r * Math.cos(a0), cy + r * Math.sin(a0)];
    const [xo1, yo1] = [cx + r * Math.cos(a1), cy + r * Math.sin(a1)];
    const [xi0, yi0] = [cx + ri * Math.cos(a1), cy + ri * Math.sin(a1)];
    const [xi1, yi1] = [cx + ri * Math.cos(a0), cy + ri * Math.sin(a0)];
    return `<path d="M${xoo.toFixed(1)},${yoo.toFixed(1)} A${r},${r} 0 ${lg},1 ${xo1.toFixed(1)},${yo1.toFixed(1)} L${xi0.toFixed(1)},${yi0.toFixed(1)} A${ri},${ri} 0 ${lg},0 ${xi1.toFixed(1)},${yi1.toFixed(1)} Z" fill="${seg.c}"/>`;
  }).join('');
  const sep = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,.7)" stroke-width="1.5"/><circle cx="${cx}" cy="${cy}" r="${ri}" fill="none" stroke="rgba(255,255,255,.7)" stroke-width="1.5"/>`;
  const center = _tx(cx, cy - 3, _fmtV(total), { size: 14, w: '700' }) + _tx(cx, cy + 12, '总量', { size: 9, fill: _C.mid });
  const legend = segs.map((seg, i) => {
    const lx = 165, ly = 20 + i * 22;
    const pct = ((Math.abs(seg.v) / total) * 100).toFixed(1);
    return `<rect x="${lx}" y="${ly - 8}" width="9" height="9" rx="2" fill="${seg.c}"/>${_tx(lx + 13, ly, seg.label, { a: 'start', size: 9.5, fill: _C.mid })}${_tx(W - 4, ly, pct + '%', { a: 'end', size: 9.5, fill: _C.ink, w: '600' })}`;
  }).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${paths}${sep}${center}${legend}</svg>`;
}

/* ── Waterfall bridge ────────────────────────────────────────── */
function _waterfall(steps, W = 520, H = 190) {
  const pL = 8, pR = 8, pT = 28, pB = 36;
  const cW = W - pL - pR, cH = H - pT - pB;
  const colW = cW / steps.length;
  const maxV = Math.max(...steps.map((s) => s.base + Math.max(0, s.delta)), ...steps.map((s) => s.base), 1);
  const toY = (v) => pT + cH - (v / maxV) * cH;
  const baseY = pT + cH;
  const gridLines = [0.25, 0.5, 0.75, 1].map((t) => {
    const y = toY(maxV * t);
    return `<line x1="${pL}" y1="${y.toFixed(1)}" x2="${W - pR}" y2="${y.toFixed(1)}" stroke="${_C.track}" stroke-width="0.7"/>` +
      _tx(pL + 1, y - 2, _fmtV(maxV * t), { a: 'start', size: 7.5, fill: _C.dim });
  }).join('');
  const elements = steps.map((step, i) => {
    const x = pL + i * colW, bx = x + colW * 0.14, bw = colW * 0.72;
    const top = Math.min(step.base, step.base + step.delta);
    const bot = Math.max(step.base, step.base + step.delta);
    const by = toY(bot), bh = Math.max(2, (bot - top) / maxV * cH);
    const fill = step.final ? _C.gold : (step.delta >= 0 ? _C.pos : _C.neg);
    const next = steps[i + 1];
    const conn = next
      ? `<line x1="${(bx + bw).toFixed(1)}" y1="${toY(step.base + step.delta).toFixed(1)}" x2="${(x + colW + colW * 0.14).toFixed(1)}" y2="${toY(next.base).toFixed(1)}" stroke="${_C.line}" stroke-width="0.8" stroke-dasharray="3,2"/>`
      : '';
    const labelV = step.final ? _fmtV(step.base + step.delta) : (step.delta >= 0 ? '+' : '') + _fmtV(step.delta);
    const labelFill = step.final ? _C.gold : (step.delta >= 0 ? _C.pos : _C.neg);
    return `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="3" fill="${fill}" opacity="${step.final ? 1 : 0.82}"/>
      ${conn}${_tx((bx + bw / 2).toFixed(1), (by - 5).toFixed(1), labelV, { size: 8, fill: labelFill, w: '700' })}
      ${_tx((bx + bw / 2).toFixed(1), (H - 8).toFixed(1), step.label, { size: 8, fill: _C.mid })}`;
  }).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible">
    ${gridLines}<line x1="${pL}" y1="${baseY}" x2="${W - pR}" y2="${baseY}" stroke="${_C.line}" stroke-width="1"/>${elements}</svg>`;
}

/* ── Simple metric bar ───────────────────────────────────────── */
function _metricBar(value, max, label, W = 150, H = 58) {
  const clamp = Math.max(0, Math.min(1, value / (max || 1)));
  const pX = 8;
  const barY = 32;
  const barH = 8;
  const barW = W - pX * 2;
  const valW = Math.max(2, barW * clamp);
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    ${_tx(pX, 13, label, { a: "start", size: 8.5, fill: _C.mid, w: "600" })}
    ${_tx(W - pX, 14, (value * 100).toFixed(1) + "%", { a: "end", size: 15, fill: _C.ink, w: "700" })}
    <rect x="${pX}" y="${barY}" width="${barW}" height="${barH}" rx="4" fill="${_C.track}"/>
    <rect x="${pX}" y="${barY}" width="${valW.toFixed(1)}" height="${barH}" rx="4" fill="${_C.gold}" opacity=".9"/>
    ${_tx(pX, H - 7, "0", { a: "start", size: 7.5, fill: _C.dim })}
    ${_tx(W - pX, H - 7, (max * 100).toFixed(0) + "%", { a: "end", size: 7.5, fill: _C.dim })}
  </svg>`;
}

/* ── Cashflow: grouped bar + cumulative line ─────────────────── */
function _cashflowSVG(yearly, W = 460, H = 200) {
  if (!yearly.length) return `<div style="color:${_C.dim};font-size:11px;padding:20px 0;text-align:center">暂无数据</div>`;
  const pL = 44, pR = 12, pT = 18, pB = 34, cW = W - pL - pR, cH = H - pT - pB;
  const n = yearly.length, colW = cW / n;
  const allV = [...yearly.map((r) => Number(r.cashflow || 0)), ...yearly.map((r) => Number(r.cumulative || 0))];
  const minV = Math.min(0, ...allV), maxV = Math.max(0, ...allV);
  const span = (maxV - minV) || 1;
  const toY = (v) => pT + cH - ((v - minV) / span) * cH;
  const zY = toY(0);
  const grid = [minV, 0, maxV].map((v) => {
    const y = toY(v);
    return `<line x1="${pL}" y1="${y.toFixed(1)}" x2="${pL + cW}" y2="${y.toFixed(1)}" stroke="${v === 0 ? _C.line : _C.track}" stroke-width="${v === 0 ? 1 : 0.6}"/>` +
      _tx((pL - 3).toFixed(1), y.toFixed(1), _fmtV(v), { a: 'end', size: 8, fill: _C.dim });
  }).join('');
  const bars = yearly.map((row, i) => {
    const cf = Number(row.cashflow || 0);
    const bx = pL + i * colW + colW * 0.14, bw = colW * 0.46;
    const by = Math.min(zY, toY(cf)), bh = Math.max(2, Math.abs(toY(cf) - zY));
    return `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="2" fill="${cf >= 0 ? _C.pos : _C.neg}" opacity=".82"/>`;
  }).join('');
  const pts = yearly.map((row, i) => `${(pL + i * colW + colW * 0.5).toFixed(1)},${toY(Number(row.cumulative || 0)).toFixed(1)}`).join(' ');
  const dots = yearly.map((row, i) => `<circle cx="${(pL + i * colW + colW * 0.5).toFixed(1)}" cy="${toY(Number(row.cumulative || 0)).toFixed(1)}" r="4" fill="${_C.gold}" stroke="white" stroke-width="1.5"/>`).join('');
  const xLabels = yearly.map((row, i) => _tx((pL + i * colW + colW * 0.5).toFixed(1), (H - 8).toFixed(1), `Y${row.year}`, { size: 9, fill: _C.mid })).join('');
  const leg = `<rect x="${pL}" y="3" width="10" height="7" rx="1.5" fill="${_C.pos}"/>${_tx(pL + 14, 9.5, '年度净流', { a: 'start', size: 8.5, fill: _C.mid })}<line x1="${pL + 68}" y1="7" x2="${pL + 80}" y2="7" stroke="${_C.gold}" stroke-width="2"/><circle cx="${pL + 74}" cy="7" r="3" fill="${_C.gold}"/>${_tx(pL + 84, 9.5, '累计', { a: 'start', size: 8.5, fill: _C.mid })}`;
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible">${grid}${bars}<polyline points="${pts}" fill="none" stroke="${_C.gold}" stroke-width="2" stroke-linejoin="round"/>${dots}${xLabels}${leg}</svg>`;
}

/* ── Bipolar horizontal profit contribution ───────────────────── */
function _bipolar(rows, W = 400) {
  const rowH = 26, pL = 90, pR = 58, pT = 20, pB = 10;
  const H = pT + rows.length * rowH + pB, cW = W - pL - pR, zX = pL + cW / 2;
  const maxAbs = Math.max(1, ...rows.map((r) => Math.abs(r.v)));
  const centerLine = `<line x1="${zX}" y1="${pT - 6}" x2="${zX}" y2="${H - pB}" stroke="${_C.line}" stroke-width="1"/>${_tx(zX, pT - 10, '0', { size: 8, fill: _C.dim })}`;
  const elements = rows.map((row, i) => {
    const y = pT + i * rowH, by = y + rowH * 0.12, bh = rowH * 0.76;
    const hw = (Math.abs(row.v) / maxAbs) * (cW / 2);
    const bx = row.v >= 0 ? zX : zX - hw;
    const fill = row.v >= 0 ? _C.pos : _C.neg;
    const valX = row.v >= 0 ? zX + hw + 4 : zX - hw - 4;
    return `<rect x="${pL - 11}" y="${(by + bh / 2 - 4).toFixed(1)}" width="8" height="8" rx="2" fill="${row.c || fill}" opacity=".8"/>
      ${_tx(pL - 15, (by + bh / 2 + 3.5).toFixed(1), row.label, { a: 'end', size: 9.5, fill: _C.mid })}
      <rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${hw.toFixed(1)}" height="${bh.toFixed(1)}" rx="2" fill="${fill}" opacity=".82"/>
      ${_tx(valX.toFixed(1), (by + bh / 2 + 3.5).toFixed(1), _fmtV(row.v), { a: row.v >= 0 ? 'start' : 'end', size: 8.5, fill, w: '600' })}`;
  }).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible">${centerLine}${elements}</svg>`;
}

/* ── Compact horizontal bar chart ─────────────────────────────── */
function _hbarChart(rows, W = 400, rowH = 24) {
  if (!rows.length) return `<div style="color:${_C.dim};font-size:11px;padding:16px 0;text-align:center">暂无数据</div>`;
  const pL = 108, pR = 52, pT = 8, pB = 8;
  const H = pT + rows.length * rowH + pB;
  const cW = W - pL - pR;
  const maxV = Math.max(1, ...rows.map((r) => Math.abs(r.v)));
  const bars = rows.map((row, i) => {
    const y = pT + i * rowH;
    const by = y + rowH * 0.18;
    const bh = rowH * 0.64;
    const bw = (Math.abs(row.v) / maxV) * cW;
    const fill = row.c || _C.gold;
    return `<rect x="${pL - 10}" y="${(by + bh / 2 - 4).toFixed(1)}" width="7" height="7" rx="2" fill="${fill}"/>
      ${_tx((pL - 14).toFixed(1), (by + bh / 2 + 3.5).toFixed(1), row.label, { a: "end", size: 9.5, fill: _C.mid })}
      <rect x="${pL}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="3" fill="${fill}" opacity=".88"/>
      ${_tx((pL + bw + 5).toFixed(1), (by + bh / 2 + 3.5).toFixed(1), row.val || _fmtV(row.v), { a: "start", size: 8.5, fill: _C.ink, w: "600" })}`;
  }).join("");
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible">${bars}</svg>`;
}

/* ── KPI value vs cost mini bar ───────────────────────────────── */
function _kpiCompareBar(value, cost, W = 120, H = 6) {
  const maxV = Math.max(value, cost, 1);
  const vW = (value / maxV) * W;
  const cW = (cost / maxV) * W;
  return `<svg width="${W}" height="${H * 2 + 6}" viewBox="0 0 ${W} ${H * 2 + 6}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="${W}" height="${H}" rx="${H / 2}" fill="${_C.track}"/>
    <rect x="0" y="0" width="${vW.toFixed(1)}" height="${H}" rx="${H / 2}" fill="${_C.gold}"/>
    <rect x="0" y="${H + 4}" width="${W}" height="${H}" rx="${H / 2}" fill="${_C.track}"/>
    <rect x="0" y="${H + 4}" width="${cW.toFixed(1)}" height="${H}" rx="${H / 2}" fill="${_C.pos}" opacity=".75"/>
  </svg>`;
}

function renderDashboard(stats) {
  const totalValue = Number(stats.total_value || 0);
  const tdc        = Number(stats.tdc || 0);
  const profit     = Number(stats.profit || 0);

  /* ── KPI strip ── */
  document.getElementById("kpi-blocks").textContent   = String(stats.blocks);
  document.getElementById("kpi-gfa").textContent      = formatGfaWan(stats.gfa);
  document.getElementById("kpi-cost").textContent     = formatMoneyYi(tdc);
  document.getElementById("kpi-revenue").textContent  = formatMoneyYi(totalValue);
  document.getElementById("eco-profit").textContent   = `${formatMoneyYi(profit)} (${(Number(stats.profit_on_cost || 0) * 100).toFixed(1)}% PoC)`;
  const scoreEl = document.getElementById("eco-composite-score");
  if (scoreEl) scoreEl.textContent = `${Number(stats.composite_score || 0).toFixed(1)} / 100`;

  const kpiCompare = document.getElementById("kpi-value-cost-bar");
  if (kpiCompare) {
    kpiCompare.innerHTML = `<div style="display:flex;align-items:center;gap:10px;margin-top:6px">
      ${_kpiCompareBar(totalValue, tdc, 100, 5)}
      <span style="font-size:9px;color:${_C.mid};line-height:1.4"><span style="color:${_C.gold}">■</span> 价值<br/><span style="color:${_C.pos}">■</span> 成本</span>
    </div>`;
  }

  const warn = document.getElementById("eco-warning");
  if (warn) {
    const parts = [stats.data_warning, assessDataReasonability(stats)].filter(Boolean);
    warn.textContent = parts.join(" · ");
  }

  const extra = document.getElementById("eco-kpi-extra");
  if (extra) {
    extra.innerHTML =
      `可售 ${formatGfaWan(stats.saleable_area || 0)} · 可租 ${formatGfaWan(stats.rentable_area || 0)} · ` +
      `剩余地价 ${formatMoneyYi(stats.residual_land_value)} · 峰值资金 ${formatMoneyYi(stats.peak_funding)} · 回正 Y${stats.payback_year || "—"}`;
  }

  /* ── Overview: 收入·成本对比 → Waterfall bridge ── */
  const revCost = document.getElementById("eco-rev-cost-bars");
  if (revCost) {
    const stack = stats.cost_stack || {};
    const direct = Number(stack.hard || 0) + Number(stack.infra || 0) + Number(stack.soft || 0) + Number(stack.contingency || 0);
    const land   = Number(stack.land || 0);
    const tax    = Number(stack.tax  || 0);
    const fin    = Number(stack.finance || 0);
    const mkt    = Number(stack.sale_marketing || 0) + Number(stack.lease_marketing || 0);
    const wfSteps = [
      { label: "总价值",  delta: totalValue, base: 0,                                     final: false },
      { label: "直接建安", delta: -direct,    base: totalValue,                             final: false },
      { label: "土地成本", delta: -land,      base: totalValue - direct,                   final: false },
      { label: "税费",    delta: -tax,       base: totalValue - direct - land,             final: false },
      { label: "融资",    delta: -fin,       base: totalValue - direct - land - tax,       final: false },
      { label: "营销",    delta: -mkt,       base: totalValue - direct - land - tax - fin, final: false },
      { label: "利润",    delta: profit,     base: 0,                                     final: true  },
    ];
    revCost.innerHTML = _waterfall(wfSteps, revCost.offsetWidth || 480, 190);
  }

  /* ── Overview: 现金流 → bar + cumulative line ── */
  const cfEl = document.getElementById("eco-cashflow-bars");
  if (cfEl) cfEl.innerHTML = _cashflowSVG(stats.yearly || [], cfEl.offsetWidth || 460, 200);

  /* ── Overview: 综合得分 → simple metric bars ── */
  const compBars = document.getElementById("eco-composite-bars");
  if (compBars) {
    const poc = Number(stats.profit_on_cost || 0);
    const pov = Number(stats.profit_on_gdv  || 0);
    const comp = stats.composite_components || {};
    const econW = Number(comp.econ_score || 0) / 100;
    compBars.innerHTML =
      `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px;margin-top:4px">` +
        _metricBar(poc,   0.6, "Profit on Cost",  150, 58) +
        _metricBar(pov,   0.4, "Profit on Value", 150, 58) +
        _metricBar(econW, 1.0, "经济维度得分",    150, 58) +
      `</div>`;
  }

  const breakdown = document.getElementById("eco-composite-breakdown");
  if (breakdown) {
    const comp = stats.composite_components || {};
    breakdown.textContent = `经济 ${Number(comp.econ_score || 0).toFixed(1)}/100 (70%) · 生态 ${Number(comp.eco_score || 0).toFixed(1)}/100 (30%)`;
  }

  /* ── Overview: 交通与公共投入 → donut + eco metrics ── */
  const publicInfraEl = document.getElementById("eco-public-infra");
  if (publicInfraEl) {
    const fnFin = stats.by_function_financial || {};
    const gCost = Number(fnFin.GROUND?.cost || 0);
    const grCost = Number(fnFin.GREEN?.cost || 0);
    const wCost  = Number(fnFin.WALKWAY?.cost || 0);
    const hCost  = Number(fnFin.HIGHWAY?.cost || 0);
    const pubCost = Number(fnFin.PUBLIC?.cost || 0);
    const totalPub = gCost + grCost + wCost + hCost + pubCost;
    const pubPct = tdc > 0 ? (totalPub / tdc * 100).toFixed(1) : "—";
    const eco = stats.eco || {};
    const infraSegs = [
      { label: "公共服务", v: pubCost, c: FUNCTION_COLORS.PUBLIC || "#8b5cf6" },
      { label: "慢行系统", v: wCost,  c: "#0ea5e9" },
      { label: "绿地",     v: grCost, c: "#059669" },
      { label: "道路",     v: hCost,  c: _C.gold },
      { label: "场地",     v: gCost,  c: "#64748b" },
    ].filter((s) => s.v > 0);
    const emptyHint = totalPub <= 0
      ? `<div style="font-size:10.5px;color:${_C.dim};margin:8px 0;padding:10px;background:${_C.track};border-radius:8px">未检测到交通/市政体块。请在 Rhino 中导出 WALKWAY、GREEN、HIGHWAY 图层，或确保方案含 PUBLIC 体块。</div>`
      : "";
    publicInfraEl.innerHTML =
      `<div style="font-size:11px;margin-bottom:8px">公共与交通投入 <b style="color:${_C.gold}">${formatMoneyYi(totalPub)}</b>，占 TDC <b>${pubPct}%</b></div>` +
      (infraSegs.length ? _donut(infraSegs, Math.min(publicInfraEl.offsetWidth || 300, 320), 140) : emptyHint) +
      `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px">` +
        [
          { k: "碳汇",     v: `${Number(eco.carbon_sink_tpy || 0).toFixed(1)}`, u: "tCO₂/年" },
          { k: "径流削减", v: `${Number(eco.runoff_reduction_m3y || 0).toFixed(0)}`, u: "m³/年" },
          { k: "微气候",   v: `${Number(eco.cooling_score || 0).toFixed(0)}`, u: "/100" },
        ].map((m) => `<div style="text-align:center;padding:8px 4px;background:rgba(0,0,0,.03);border-radius:8px"><div style="font-size:9px;color:${_C.mid};margin-bottom:3px">${m.k}</div><div style="font-size:14px;font-weight:700;color:${_C.ink}">${m.v}</div><div style="font-size:8.5px;color:${_C.dim}">${m.u}</div></div>`).join("") +
      `</div>`;
  }

  /* ── Mix: 功能结构 → donut ring ── */
  const fnEntries = Object.entries(stats.by_function || {});
  const mix = document.getElementById("kpi-by-function");
  if (mix) {
    const segs = fnEntries
      .filter(([, v]) => Number(v) > 0)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .map(([name, v]) => ({ label: name.replace(/_/g, " "), v: Number(v), c: FUNCTION_COLORS[name] || _C.mid }));
    mix.innerHTML = _donut(segs, mix.offsetWidth || 320, 175);
  }

  /* ── Mix: 分功能收入贡献 ── */
  const revBars = document.getElementById("eco-mix-revenue-bars");
  if (revBars) {
    const rows = fnEntries
      .map(([name]) => ({ label: name.replace(/_/g, " "), v: Number(stats.by_function_financial?.[name]?.revenue || 0), c: FUNCTION_COLORS[name] || _C.mid }))
      .filter((r) => r.v > 0).sort((a, b) => b.v - a.v)
      .map((r) => ({ ...r, val: formatMoneyYi(r.v) }));
    revBars.innerHTML = _hbarChart(rows, revBars.offsetWidth || 400);
  }

  /* ── Mix: 利润贡献 → bipolar ── */
  const costBars = document.getElementById("eco-mix-cost-bars");
  if (costBars) {
    const head = costBars.previousElementSibling;
    if (head?.classList.contains("dash-card-head")) head.textContent = "利润贡献分析";
    const rows = fnEntries
      .map(([name]) => ({ label: name.replace(/_/g, " "), v: Number(stats.by_function_financial?.[name]?.profit || 0), c: FUNCTION_COLORS[name] || _C.mid }))
      .sort((a, b) => b.v - a.v);
    costBars.innerHTML = _bipolar(rows, costBars.offsetWidth || 380);
  }

  /* ── Mix: 收益效率 ── */
  const effBars = document.getElementById("eco-mix-efficiency-bars");
  if (effBars) {
    const rows = fnEntries
      .map(([name]) => {
        const fin = stats.by_function_financial?.[name] || {};
        const gfa = Number(fin.gfa || 0);
        const p   = Number(fin.profit || 0);
        return { label: name.replace(/_/g, " "), v: gfa > 0 ? p / gfa : 0, c: FUNCTION_COLORS[name] || _C.mid };
      })
      .filter((r) => r.v !== 0)
      .sort((a, b) => b.v - a.v)
      .map((r) => ({ ...r, val: `${Number(r.v).toFixed(0)} 元/㎡` }));
    effBars.innerHTML = _hbarChart(rows, effBars.offsetWidth || 400);
  }

  /* ── Mix: 决策敏感性 → metric bars + driver table ── */
  const sens = document.getElementById("eco-sensitivity");
  if (sens) {
    const poc = Number(stats.profit_on_cost || 0);
    const pov = Number(stats.profit_on_gdv  || 0);
    const drivers = [
      { k: "住宅售价",    impact: "高" },
      { k: "办公/商业租金", impact: "高" },
      { k: "Cap Rate",   impact: "高" },
      { k: "土地成本",    impact: "中" },
      { k: "融资利率",    impact: "中" },
    ];
    sens.innerHTML =
      `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px;margin-bottom:10px">` +
        _metricBar(poc, 0.6, "Profit on Cost",  150, 58) +
        _metricBar(pov, 0.4, "Profit on Value", 150, 58) +
      `</div>` +
      `<div style="font-size:10px;color:${_C.mid};font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px">关键敏感因子</div>` +
      drivers.map((d) => `<div style="display:flex;justify-content:space-between;font-size:11px;padding:4px 0;border-bottom:1px solid ${_C.track}"><span style="color:${_C.mid}">${d.k}</span><span style="color:${d.impact === "高" ? _C.neg : _C.gold};font-weight:600">${d.impact}敏感</span></div>`).join("") +
      `<div style="margin-top:8px;font-size:10.5px;color:${_C.dim}">5年利润为未折现口径，不等同于 IRR。</div>`;
  }

  /* ── Mix: 方案雷达图 ── */
  const radar = document.getElementById("eco-radar");
  if (radar) {
    const margin = totalValue > 0 ? profit / totalValue : 0;
    const axes = [
      { k: "规模",   v: Math.min(1, Number(stats.gfa || 0) / 800000) },
      { k: "价值",   v: Math.min(1, totalValue / 3e10) },
      { k: "利润率", v: Math.max(0, Math.min(1, margin + 0.35)) },
      { k: "PoC",    v: Math.max(0, Math.min(1, Number(stats.profit_on_cost || 0) + 0.35)) },
      { k: "资金",   v: Math.max(0, Math.min(1, 1 - Number(stats.peak_funding || 0) / Math.max(1, tdc))) },
    ];
    const rW = radar.offsetWidth || 260, rH = 200, cx = rW / 2, cy = 100, r0 = 70;
    const pts = axes.map((a, i) => {
      const ang = (-90 + i * 72) * Math.PI / 180;
      return [cx + Math.cos(ang) * r0 * a.v, cy + Math.sin(ang) * r0 * a.v];
    });
    const poly = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const rings = [1, 0.66, 0.33].map((s) => {
      const rPts = axes.map((_, i) => { const a = (-90 + i * 72) * Math.PI / 180; return `${(cx + Math.cos(a) * r0 * s).toFixed(1)},${(cy + Math.sin(a) * r0 * s).toFixed(1)}`; }).join(" ");
      return `<polygon points="${rPts}" fill="none" stroke="${_C.track}" stroke-width="${s === 1 ? 1 : 0.6}"/>`;
    }).join("");
    const spokes = axes.map((a, i) => {
      const ang = (-90 + i * 72) * Math.PI / 180;
      const sx = cx + Math.cos(ang) * r0, sy = cy + Math.sin(ang) * r0;
      const lx = cx + Math.cos(ang) * (r0 + 16), ly = cy + Math.sin(ang) * (r0 + 16);
      return `<line x1="${cx}" y1="${cy}" x2="${sx.toFixed(1)}" y2="${sy.toFixed(1)}" stroke="${_C.line}" stroke-width="1"/>` +
        _tx(lx.toFixed(1), (ly + 3).toFixed(1), a.k, { size: 9.5, fill: _C.mid }) +
        `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="2" fill="${_C.dim}"/>`;
    }).join("");
    radar.innerHTML = `<svg width="${rW}" height="${rH}" viewBox="0 0 ${rW} ${rH}" xmlns="http://www.w3.org/2000/svg">
      ${rings}${spokes}
      <polygon points="${poly}" fill="${_C.goldM}" stroke="${_C.gold}" stroke-width="2"/>
      ${pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.5" fill="${_C.gold}" stroke="white" stroke-width="1.2"/>`).join("")}
    </svg>`;
  }

  /* ── Formula & notes ── */
  const formula = document.getElementById("eco-formula");
  if (formula) {
    const stack = stats.cost_stack || {};
    formula.innerHTML =
      `<div style="margin-bottom:5px"><b>GDV</b> = Σ(可售面积 × 销售单价)</div>` +
      `<div style="margin-bottom:5px"><b>资本化价值</b> = Σ(可租面积 × 租金 × 出租率 × (1−OPEX) / Cap Rate)</div>` +
      `<div style="margin-bottom:5px"><b>TDC</b> = 建安 ${formatMoneyYi(stack.hard)} + 市政 ${formatMoneyYi(stack.infra)} + 软成 ${formatMoneyYi(stack.soft)} + 预备 ${formatMoneyYi(stack.contingency)} + 土地 ${formatMoneyYi(stack.land)} + 税费 ${formatMoneyYi(stack.tax)} + 营销 ${formatMoneyYi((Number(stack.sale_marketing || 0) + Number(stack.lease_marketing || 0)))} + 融资 ${formatMoneyYi(stack.finance)}</div>` +
      `<div><b>利润</b> = 总价值 − TDC = <b style="color:${_C.gold}">${formatMoneyYi(profit)}</b> · 回正 Y${stats.payback_year || "—"} · 峰值资金 ${formatMoneyYi(stats.peak_funding)}</div>`;
  }

  const sensitivityNotes = document.getElementById("eco-sensitivity-notes");
  if (sensitivityNotes) {
    sensitivityNotes.innerHTML =
      `<div style="margin-bottom:5px"><b>保守版：</b><span style="color:${_C.mid}">弱市场与高成本情景，上调土地/融资/税费，下调售价/租金/出租率，放宽 Cap Rate。</span></div>` +
      `<div style="margin-bottom:5px"><b>基准版：</b><span style="color:${_C.mid}">常规汇报口径，中性土地与融资条件，价格与租值取中位判断。</span></div>` +
      `<div style="margin-bottom:5px"><b>乐观版：</b><span style="color:${_C.mid}">上行情景，更强去化、更高租金与出租率、更紧 Cap Rate，不建议单独作为决策依据。</span></div>` +
      `<div><b>敏感性顺序：</b><span style="color:${_C.mid}">住宅售价 → 办公/商业租金 → Cap Rate → 土地成本 → 融资利率</span></div>`;
  }

  const notes = document.getElementById("eco-assumption-notes");
  if (notes) {
    notes.innerHTML = `<span style="color:${_C.mid}">① 前端面板是唯一经济参数源。② TDC 含土地、税费、营销、融资。③ GREEN/GROUND/WALKWAY/HIGHWAY 计成本不计市场收入。④ 三套预设用于区间判断。</span>`;
  }
}
