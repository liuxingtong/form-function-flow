# Form Function Flow

本仓库现已切换为“基于真实原始观测数据”的分析流程。

八时段切片、POI 分时段合成、分时段 proxy 校准、synthetic flow 等旧链路已经归档到 `_deprecated/`，不再作为当前分析入口使用。

## 当前有效的数据路径

- 静态空间底图：`data/site_3km/01_units.gpkg`、`data/site_3km/02_edges.csv`
- 站域 CAD 与场地红线（WGS84）：
  - 场地红线：`data/SITE.json`
  - `site.dxf` / `district.dxf` 按图层 GeoJSON：`data/site_dxf/`、`data/district_dxf/`（`local/` 为 CAD 坐标，`crs84/` 为与 `SITE.json` 对齐后的经纬度；脚本见 `scripts/align_cad_geojson_to_SITE_json.py`，参数见 `data/cad_alignment/transform_meta.json`）
- 真实动态观测：
  - `data/site_3km/metroflow/inout_10min_3km.parquet`
  - `data/site_3km/metroflow/od_internal_10min_3km.parquet`
  - 其他带真实时间戳的 LBS / 客流 / OD 原始表
- SSHMM 观测构建：
  - `scripts/build_sshmm_observations.py`
  - `scripts/export_sshmm_o_npy.py`

## 当前建议工作流

1. 用 `scripts/build_site_units_and_edges.py` 维护研究单元与边关系。
2. 用真实带时间戳的数据构建 `unit_id x timestamp` 观测长表。
3. 用 `scripts/build_sshmm_observations.py` 生成：
   - `observation_long.csv`
   - `observation_normalized.csv`
   - `sshmm_manifest.json`
4. 在此基础上做原始时序分析、状态识别或地块分区判断。

详细说明见：

- [原始观测分析流程](docs/raw-observation-workflow.md)
- [基于分析结果的 site 地块功能分区操作](docs/site-zoning-playbook.md)

## 归档说明

- `_deprecated/data/`：旧的八时段代理数据与相关状态表
- `_deprecated/output/`：旧的八时段结果和 synthetic flow 输出
- `_deprecated/docs/`：旧的八时段方法文档与参考整理
