# 根据手册截图可推断的 CSV 字段（与 gdb 图层对应）

以下为「百度 LBS / GIS 平台」说明里出现的图层与字段语义。**实际导出列名可能是中英文混合**，平台升级也可能微调名称；`scripts/build_sshmm_observations.py` 会用候选名自动匹配（见脚本内 `_pick_col` / `consume_od_points`）。

---

## `point_flow_day`（日粒度客流）

| 语义 | 手册提及 / 常见列名 |
|------|---------------------|
| 日期 | `日期` / `date` |
| 网格 | `网格 ID` / `gridID` |
| 坐标 | `x 坐标`/`x`、`y 坐标`/`y`（WGS84） |
| 人数 | `人数` / `count` |

---

## `point_flow_hour`（小时粒度客流）— **构造 SSHMM 时间序列的主输入之一**

| 语义 | 手册提及 / 常见列名 |
|------|---------------------|
| 时刻（到小时） | `时间` / `time` / `timestamp` |
| 网格 | `网格 ID` / `gridID`（可选；**有点坐标则可不配**） |
| 坐标 | `x`、`y` 或 `x 坐标`、`y 坐标` |
| 人数 | `人数` / `count` |

---

## `point_OD_hour_O`（起点在郊环内的小时 OD）

用于汇总 **从某点出发** 的流量 → 脚本映射为单元 **`o_leave`**。

| 语义 | 手册提及 / 常见列名 |
|------|---------------------|
| 时间 | `时间` / `time` |
| 起点坐标 | `JobX` `JobY`、`HousingX` `HousingY`，或 `ox`/`oy` |
| 人数 | `数量` / `人数` / `count` |

---

## `point_OD_hour_D`（终点在郊环内的小时 OD）

用于汇总 **到达某点** 的流量 → **`o_arrive`**。

| 语义 | 常见列名 |
|------|----------|
| 时间 | `时间` / `time` |
| 终点坐标 | `dx`/`dy`，或终点相关 `JobX`/`HousingX` 等（与平台定义一致） |
| 人数 | `数量` / `count` |

---

## `point_housing` / `point_job`（职住 OD，**年/季度**）

不是小时序列本体；若只有这类表，**无法直接替代** `point_flow_hour` + 小时 OD，除非自行按假设拆到小时（不推荐冒充真实数据）。

---

## `point_resident` / `point_portrait`

| 图层 | 用途 |
|------|------|
| `point_resident` | 网格 + 人口类型 + 人数 → 可与静态人口分布对照 |
| `point_portrait` | 性别、年龄、收入、行业等 → **解释**功能/人群，**不是**论文 Definition 1 里的 POI 签到向量 |

论文所需的 **`o_{4..L}` 语义维**：若无签到，用 **`poi_static`（每 unit 各业态计数）** 替代（脚本会按小时复制）。

---

## 最小可跑组合（无真实导出时）

1. **`--demo`**：完全不需要 CSV（见 `build_sshmm_observations.py --demo`）。  
2. **合成源表**：见 `sample_inputs/` 下由 `write_sshmm_synthetic_source_csvs.py` 生成的文件，字段对齐上表，坐标落在研究范围内。
