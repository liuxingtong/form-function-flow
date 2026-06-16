# Phase 2 Financial Modeling Framework (for Scenario Blocks)

This document defines a practical, auditable framework for early-stage scenario finance.

## 1) Modeling Principle

Use a **concept-level development pro forma**:

- fast enough for design iteration,
- structured enough for investment logic,
- parameterized for later calibration to local market data.

Core outputs:

- per-use development volume
- total and phased costs
- revenue and operating cash flow
- project return metrics (IRR / ROI / payback)

## 2) Recommended Capital and Cash-Flow Structure

Use a two-stage structure:

1. **Development stage (t0~tn construction)**
   - land / demolition / preparation
   - hard + soft + contingency
   - financing costs during construction

2. **Operation & exit stage (stabilization + hold)**
   - NOI generation from retained assets
   - optional sale/exit valuation at terminal cap rate

Suggested return views:

- unlevered project IRR
- levered equity IRR
- equity multiple
- simple ROI
- payback period

## 3) Cost Breakdown (parameterized)

## 3.1 Hard costs

- superstructure, facade, MEP, fit-out baseline
- external works / landscape / municipal connection
- over-track structural premium (TOD deck only)

## 3.2 Soft costs

- design / consultant / PM
- permits / approvals / legal
- marketing / leasing / sales expense
- taxes and transactional costs (project-localized)

## 3.3 Financing

- construction interest (drawdown based)
- loan fees
- DSCR/LTV/LTC constraints for debt sizing

## 3.4 Contingency

- as percent of hard cost (e.g., 8-12% initial range, adjustable)

## 4) Revenue Logic by Use Type

For each block:

- `saleable_gfa` and `retained_gfa`
- sales price path (if sell)
- rent + occupancy + growth assumptions (if hold)

Recommended by zone intent:

- **CBD**: office + retail rent-led model (higher stabilized NOI weight)
- **OFC**: office/creative + event/culture support (more conservative rent-up)
- **TOD_DECK**: residential sell-through as primary cash source + supporting retail/public package
- **TOD_GROUND**: small commercial pods, low capex but moderate turnover uncertainty

## 5) Core Formulas (MVP)

- `gross_revenue_sale = saleable_gfa * unit_sale_price`
- `gross_revenue_rent_t = rentable_gfa * rent_t * occupancy_t`
- `effective_gross_income_t = gross_revenue_rent_t + other_income_t`
- `NOI_t = effective_gross_income_t - opex_t - reserve_t`
- `total_dev_cost = land + hard + soft + financing + contingency`
- `project_value_exit = NOI_stabilized / exit_cap_rate`
- `project_profit = total_inflows - total_outflows`

Returns:

- `IRR` from full-period cash flow
- `ROI = project_profit / total_equity_invested`
- `payback_year = first year cumulative_equity_cf >= 0`

## 6) Zone-Specific Adjustment Factors (initial defaults)

These are placeholders to be calibrated with local market data.

- `cost_multiplier`:
  - CBD: 1.00-1.10
  - OFC: 0.95-1.05
  - TOD_GROUND: 0.85-1.00
  - TOD_DECK: 1.10-1.30 (over-track premium)
- `rent_up_speed`:
  - CBD faster than OFC/TOD_GROUND
- `sales_velocity`:
  - TOD_DECK residential sensitive to absorption assumptions

## 7) Minimal Input Schema (for editor -> finance engine)

Per block:

- `block_id`
- `zone_id`
- `use_type`
- `footprint_sqm`
- `gfa_sqm`
- `height_m`
- `floors`
- `sale_ratio` (0-1)
- `cost_profile_id`
- `revenue_profile_id`

Global scenario parameters:

- financing: debt ratio, rate, tenor, fees
- schedule: start year, construction years, hold years
- market: rent growth, cap rate, vacancy bands
- policy: tax/fee assumptions

## 8) Output Tables (must-have)

1. **Use program summary**
   - GFA by use and zone
2. **Cost summary**
   - hard/soft/finance/contingency by phase
3. **Revenue summary**
   - sales + NOI by year
4. **Returns**
   - IRR/ROI/equity multiple/payback
5. **Sensitivity**
   - +/-10% price, +/-10% cost, cap rate shocks

## 9) Data Governance and Traceability

For each run, persist:

- input snapshot
- parameter version
- output tables
- rule/assumption notes

This allows comparing scenario A/B/C reproducibly.

## 10) External Reference Pointers

- ITDP TOD framework for land-use/transit integration:
  <https://tod.itdp.org/tod-standard/tod-standard-framework.html>
- ULI development finance and pro forma resources:
  <https://americas.uli.org/finance-for-real-estate-development-tools/>
  <https://americas.uli.org/professional-real-estate-development-developers-tool-kit/>
- Practical cost structure discussions (hard/soft/contingency benchmarks, to be localized):
  <https://blog.iq.dwellsy.com/hard-costs-vs-soft-costs-a-real-estate-development-budget-guide/>

