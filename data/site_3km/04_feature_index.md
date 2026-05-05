# 第 0 步字段索引（上海站 3 km 样例底板）

本文档对应 `data/site_3km` 下已生成的空间单元、边表与时间切片产物。**单位与负责人**请在组内补全后随迭代更新。

---

## `01_units.gpkg`（图层 `units`，CRS EPSG:4326）

| 字段 | 类型 | 单位 / 取值 | 来源与算法 | 备注 |
|------|------|----------------|------------|------|
| `unit_id` | string | 唯一键 | `plot_` + 原始地块 `id` | 与全表主键对齐 |
| `geometry` | Polygon | WGS84 度 | 上海地块 `plot_84_51N` ∩ `SITE.buffer_3km` | 原始几何 CRS 以 `data/all/上海地块/plot_84_51N.prj` 为准再裁切 |
| `area` | float | m² | 在投影坐标系下 `geometry.area`（与 prj 一致） | 非椭球面积 |
| `centroid_x` | float | 经度 ° | 质心转 WGS84 | |
| `centroid_y` | float | 纬度 ° | 质心转 WGS84 | |
| `dist_to_station` | float | m | 质心至上海火车站参考点（MetroFlow 默认中心）的平面距离 | 与 `metroflow/manifest_metroflow.json` 中 `center_wgs84` 一致 |
| `ring_zone` | string | `0_1km` / `1_3km` / `3_4km` | 由 `dist_to_station` 分环 | 缓冲半径 3 km 时外围环可能极少 |
| `PLOTNUMBER` 等 | 多种 | — | 源 shapefile 属性 | 可选保留 |

**复算脚本**：`scripts/build_site_units_and_edges.py`

---

## `02_edges.csv`

有向边：对每个无向邻接对 `(A,B)` 写入 `(A→B)` 与 `(B→A)` 两行，便于按 `source_id` 做行归一化。

| 字段 | 类型 | 单位 | 来源与算法 |
|------|------|------|------------|
| `source_id` | string | — | 起点 `unit_id` |
| `target_id` | string | — | 邻接单元 `unit_id` |
| `edge_kind` | string | `parcel_touch` / `knn_bridge` / **`proximity_bridge`** | **相接**；**knn**：touch 图中度为 0 时连最近邻；**proximity**：质心在投影坐标下距离 ≤ `proximity_radius_m` 且非 touch/knn 对，按距离升序贪心加入，每 `unit_id` 至多 `proximity_max_per_node` 条，用于削弱「被路缝隔开」的图隔离（弱于 touch，conductance 乘 `proximity_conductance_mult`） |
| `shared_length_m` | float | m | 相接边界交线长度；`knn_bridge` 为 0 |
| `centroid_dist_m` | float | m | 两质心在 **地块 prj 投影** 下的距离（与 `build_site_units_and_edges.py` 一致） |
| `walk_dist_m` | float | m | `centroid_dist_m × tortuosity`（默认 1.22） |
| `walk_time_min` | float | min | `walk_dist_m / (5000/60)`，按 5 km/h 步行 | **v1**：未接 OSM 步行网 |
| `cross_arterial` | int | 0/1 | 质心连线是否与快速路缓冲面相交 |
| `cross_rail` | int | 0/1 | 质心连线是否与铁路线缓冲面相交 |
| `cross_river` | int | 0/1 | 质心连线是否与水系缓冲面相交 |
| `cross_elevated` | int | 0/1 | **v1 恒 0**（缺独立高架中心线层） |
| `has_crossing_facility` | int | 0/1 | **v1 恒 0**（缺过街设施矢量） |
| `crossing_type` | string | — | **v1 空字符串** |
| `barrier_cost` | float | — | `λ1×cross_arterial + λ2×cross_rail + λ3×cross_river`（脚本内常数） |
| `angular_cost` | float | — | **v1 恒 0** |
| `edge_cost` | float | — | `α×walk_time_min + barrier_cost`（与 `形+功+流状态场.md` §0.4 一致骨架） |
| `edge_conductance` | float | — | `exp(-θ × edge_cost)`，默认 θ=0.12；**`knn_bridge` 再乘以** `bridge_conductance_mult`（见 `build_units_edges_meta.json`） |
| `edge_weight_norm` | float | — | 对每个 `source_id`，`edge_conductance / Σ邻居 conductance` |

**屏障图层（裁至 3 km 掩膜）**：`上海市_铁路线.geojson`、`上海市_水系-开源.geojson`、`上海市_城市快速路.geojson`。

**参数与说明**：`data/site_3km/build_units_edges_meta.json`。

**局限与缓解（影响下游时必读）**

| 问题 | 对下游的影响 | 缓解 |
|------|----------------|------|
| 仅 `parcel_touch` 时大量度为 0 / 路缝两侧无共边 | 图传播、`edge_weight_norm` 邻域加权断连或偏弱 | 默认加 `knn_bridge` + **`proximity_bridge`**（可调 `--proximity-radius-m` 等）；纯物理图用 `--no-proximity-bridges` / `--no-bridge-isolated` |
| 质心直线步行 | 运行可达性、瓶颈解释偏差 | OSM 步行网替换 `walk_*` 后重算边权 |
| 质心线判屏障 | 漏检绕行仍跨障的路径 | 增加「共享边界 ∩ 缓冲层」判定；补过街/高架数据 |
| 地块≠街坊 | 与文档理想单元定义不一致 | 替换单元图层后重跑建边脚本 |

---

## `data/site_3km/qa/`（质量检查图）

运行 `scripts/visualize_units_edges_qa.py` 生成：`units_ring_zones.png`、`units_edges_touch_vs_bridge.png`（灰=相接、蓝=proximity、红=knn）、`edges_choropleth_conductance.png`、`edges_choropleth_edge_cost.png`（色标上限默认 p98，可用 `--edge-color-pct`）、`degree_histogram.png`、`unit_area_histogram.png`、`qa_summary.json`；以及时序图 `temporal_wd_we_curves.png`、`temporal_slice_mass_wd_we.png`、`temporal_flow_inflow_weight_weekday.png`（若存在 `metroflow/time_slice_calibration.json` 则另有 `temporal_metroflow_blend_mass_weekday.png`）。地图线段质心为 **EPSG:32651 质心转 WGS84**。

---

## `03_time_slices.csv` 与 `poi_temporal_synthesis.json`

见 `docs/形+功+流状态场.md` §0.3；复算：`scripts/site_3km_poi_temporal_synthesis.py`。

---

## `metroflow/time_slice_calibration.json`（可选）

运行层时段权重地铁校准；见 §0.3 与 `scripts/calibrate_timeslices_from_metroflow.py`。

---

## 待补数据（不影响当前文件存在性）

- 过街设施、高架独立中心线：用于 `has_crossing_facility`、`cross_elevated` 与更可信的 `cross_arterial`。
- OSM / 步行网络：替换 `walk_dist_m` / `walk_time_min` 的质心直线近似。
- 各分析成员产出字段（`morph_state.csv`、`func_state.csv` 等）：在各自章节追加本表或子表。
