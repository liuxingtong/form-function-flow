# MetroFlow 研究区子集（上海站 3km）

原始数据：`data/all/MetroFlow/MetroFlow/`（Scientific Data / Figshare，2017-05～08，论文见 `https://doi.org/10.1038/s41597-025-05416-8`）。

## 易错点：`station` 列是 **stationID**

`metroData_InOutFlow.csv` 的 `station` 以及 `metroData_ODFlow.csv` 的 `originstation` / `destinationstation` 与 **`stationInfo.csv` 的 `stationID` 列**一致，**不是** `stationInfo` 表最左侧无名列（pandas 行号）。

子集脚本已按 **stationID** 筛选。

## 已生成的兼容产物（CRS84 / EPSG:4326）

| 文件 | 说明 |
|------|------|
| `stations_3km.geojson` | 3km 缓冲内 23 个站点；`properties.stationID` 与 Parquet 的 `station` 对齐 |
| `inout_10min_3km.parquet` | 上述站点 10 分钟进出站流（约 2.9×10⁵ 行） |
| `od_internal_10min_3km.parquet` | 起讫站**均在**缓冲内的 OD（约 2.8×10⁶ 行） |
| `MetaData_workday_calendar.csv` | 是否工作日 |
| `MetaData_shanghai_weatherHourly.csv` | 小时天气 |
| `manifest_metroflow.json` | 字段说明、行数、中心点与缓冲半径 |
| `time_slice_calibration.json` | 将 10min 进出站按 `03_time_slices` 四小时窗 + `MetaData_workday_calendar` 池化后的**经验** `flow_proxy`，及与 `poi_temporal_synthesis.json` 的**可选凸组合**（默认 α=0.5） |

## 时序校准（与 POI 合成一致的四窗）

```bash
python scripts/calibrate_timeslices_from_metroflow.py
# 仅信 MetroFlow 形状：--blend-alpha 1.0；仅保留 POI：--blend-alpha 0
python scripts/calibrate_timeslices_from_metroflow.py --blend-alpha 0.5
```

产物中的 `flow_proxy_period_weights_blended` 结构与 `poi_temporal_synthesis.json` 的 `flow_proxy_period_weights` 相同，下游可整段替换读取；经验分向权重由真实进站/出站四窗总量相对四窗均值得到（与 POI 脚本里「反转曲线」启发式不同，见该 JSON 内 `note`）。

## 复算命令

```bash
pip install duckdb pandas
python scripts/build_metroflow_site_subset.py
# 跳过 12GB OD（仅站点 + inout）：
python scripts/build_metroflow_site_subset.py --skip-od
```

默认中心为 **上海火车站** `stationID=2034` 坐标；缓冲默认 3000 m，与 `site_3km` 常用尺度一致。可改：`--center-lon`、`--center-lat`、`--buffer-m`。

## 与 `01_units.gpkg` / 其它图层对齐

1. **空间**：站点为 WGS84 点；将 `stations_3km.geojson` 与 `unit` 面做 `sjoin`（或缓冲到格网），把各站客流按权重分摊到 `unit_id`（简单做法：最近邻或落在站内）。
2. **时间**：`date` + `timeslot` + `starttime`/`endtime` 对应 10 分钟槽；与项目 `03_time_slices.csv` 对齐时，将 10 分钟聚合到你们的四个 `t_id` 小时窗（对 `inflow`/`outflow` 求和即可）。
3. **日历**：用 `MetaData_workday_calendar.csv` 区分工作日/周末，再分别聚合或校准合成曲线。
4. **年份**：数据为 **2017**；与 2025 POI 叠用时仅作**节律形状**校准，不作绝对量对标。

## 读取示例（Python）

```python
import geopandas as gpd
import pandas as pd

stations = gpd.read_file("data/site_3km/metroflow/stations_3km.geojson")
flow = pd.read_parquet("data/site_3km/metroflow/inout_10min_3km.parquet")
# flow["station"] == stations["stationID"]
```
