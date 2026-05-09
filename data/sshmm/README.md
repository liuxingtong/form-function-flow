# SSHMM 观测构造（百度 LBS / gdb 导出 → 论文 Definition 1）

本目录存放 **State-sharing HMM** 论文所需的 **Mobility Behaviour Observation** 长表及元数据。流水线脚本：`scripts/build_sshmm_observations.py`。

## 论文对齐要点

- **观测**：每个区域 \(r\)、每个时间槽 \(n\)，长度 \(L\) 的向量  
  - \(o_{r,n,1..3}\)：到达 / 离开 / 停留（原文为聚合人流；此处口径见脚本参数 `--stay-mode`）  
  - \(o_{r,n,4..L}\)：各类 POI「语义」强度（原文为签到频次；无签到时用 **TF-IDF(静态 POI 或替代列)** 或外接品类计数）
- **归一化（§4.1）**  
  - 人流三维：对每个区域 \(r\)，在**时间维**上做 min–max → \([0,1]\)  
  - 语义维：先在**每个时间槽**内对「区域 × 品类」矩阵做 **TF-IDF**，再对每个区域在**时间维** min–max

## 输入文件约定（与平台导出字段对应）

脚本接受 **CSV**（从 `DATA_shanghai_hm.gdb` 或平台打包图层导出）。列名不绑死，可用参数映射。

### 1. `point_flow_hour`（网格小时客流）

| 语义 | 建议列名（任选其一） |
|------|----------------------|
| 时间 | `time`, `timestamp`, `时间` |
| 网格或坐标 | `gridID` + `x`,`y` **或** 仅 `x`,`y`（WGS84，与手册一致） |
| 人数 | `count`, `人数` |

用于 **`stay` 或 present**  proxy（见 `--stay-mode`）。

### 2. `point_OD_hour_O`（起点在域内的小时 OD）

| 语义 | 列 |
|------|-----|
| 时间 | `time` / `timestamp` / `时间` |
| 起点坐标 | `ox`,`oy` 或 `JobX`,`JobY` / `HousingX`,`HousingY`（按手册实际字段映射） |
| 流量 | `count` / `数量` / `人数` |

聚合为各 **`unit_id` 的离开量（outflow）**。

### 3. `point_OD_hour_D`（终点在域内的小时 OD）

| 语义 | 列 |
|------|-----|
| 终点坐标 | `dx`,`dy` 或手册中终点 x/y |
| 其余同 O | |

聚合为各 **`unit_id` 的到达量（inflow）**。

### 4. 语义维 \(o_{4..L}\)（可选）

**优先**：若有时段品类表 `poi_hour.csv`，列为 `unit_id`（或可映射网格）、`time`、各品类列或 `category`,`count`。

**否则**：提供 **静态** `poi_static_units.csv`：`unit_id` + 各品类计数（由 `02-POI&AOI` 与 `01_units` 叠加统计得到）。脚本将该静态向量 **复制到每一小时**（动态语义弱于原文，需在论文方法中声明）。

## 输出文件（默认写入 `--out-dir`）

| 文件 | 说明 |
|------|------|
| `observation_long.csv` | `unit_id`, `timestamp`, `hour`, `o_arrive`, `o_leave`, `o_stay`, `poi_cat_*`（原始正值） |
| `observation_normalized.csv` | 同上 + `norm_*` 列，论文 §4.1 归一化后 \([0,1]\) |
| `sshmm_manifest.json` | \(L\)、时间范围、`stay_mode`、输入路径与 CRS |

归一化表可直接喂入自研或官方 SSHMM 训练脚本（需将同一 `unit_id` 按时间排序成 \(O_r\)）。

## 快速开始

```bash
# 演示：根据 01_units 生成可跑的合成观测（含 9 维语义 proxy）
python scripts/build_sshmm_observations.py --demo --demo-days 7 --out-dir data/sshmm/out_demo

# 真实数据：指定导出 CSV 与列映射（示例）
python scripts/build_sshmm_observations.py ^
  --units data/site_3km/01_units.gpkg ^
  --flow-hour path/to/point_flow_hour.csv ^
  --od-o path/to/point_OD_hour_O.csv ^
  --od-d path/to/point_OD_hour_D.csv ^
  --poi-static path/to/poi_static_units.csv ^
  --out-dir data/sshmm/out_run
```

未提供 OD 时，到达/离开可为 0；未提供 flow_hour 时，`stay` 需依赖 `--stay-mode od_balance` 或其它逻辑。

## 无 gdb 导出时：手册字段对齐的合成 CSV

截图中的图层字段说明见 **`fields_from_manual.md`**。可直接生成与手册列名相近的合成源表（坐标取自真实地块质心，便于落入 `01_units`）：

```bash
python scripts/write_sshmm_synthetic_source_csvs.py --hours 48 --n-units 5
```

产物写入 `sample_inputs/`：`synthetic_point_flow_hour.csv`、`synthetic_point_OD_hour_O.csv`、`synthetic_point_OD_hour_D.csv`、`synthetic_poi_static_per_unit.csv`。再接入流水线：

```bash
python scripts/build_sshmm_observations.py ^
  --units data/site_3km/01_units.gpkg ^
  --flow-hour data/sshmm/sample_inputs/synthetic_point_flow_hour.csv ^
  --od-o data/sshmm/sample_inputs/synthetic_point_OD_hour_O.csv ^
  --od-d data/sshmm/sample_inputs/synthetic_point_OD_hour_D.csv ^
  --poi-static data/sshmm/sample_inputs/synthetic_poi_static_per_unit.csv ^
  --out-dir data/sshmm/out_synthetic_from_csv
```

示例输出（便于核对）：`data/sshmm/out_synthetic_from_csv/`。

## 对接官方 [SSHMM](https://github.com/XTxiatong/SSHMM) 训练代码

论文作者实现为 **PySpark + YARN** 的 `code/2training_spark_SSHMM.py`：从单个 NumPy 文件读入观测，形状为 **`(R, T, L)`**（`R` 个区域 × `T` 个时间槽 × `L` 维特征），与本文档的 `observation_normalized.csv` 长表等价。

### 1. 生成本仓库的 `.npy`

在已有 `observation_normalized.csv` 与 `sshmm_manifest.json` 的目录上执行：

```bash
python scripts/export_sshmm_o_npy.py ^
  --normalized-csv data/sshmm/out_demo/observation_normalized.csv ^
  --manifest data/sshmm/out_demo/sshmm_manifest.json ^
  --out-npy data/sshmm/out_demo/o_hours.npy
```

会写出：

| 文件 | 说明 |
|------|------|
| `o_hours.npy` | `O` 数组，`O[r,t,:]` 为区域 `r`、时刻 `t` 的 `L` 维向量（默认用 `norm_*` 列，顺序与 manifest 一致：到离驻 + 各 `poi_cat_*`） |
| `o_hours.meta.json` | `unit_ids`、时间轴 ISO 字符串、`feature_cols`，便于回连空间单元与作图 |

缺测的 `(unit_id, timestamp)` 在长表中不存在时，默认填 **0**；可用 `--fill-missing nan` 改为 NaN（若你自改训练代码做掩码）。

### 2. 与作者脚本的差异（接入前必读）

- **路径与 API**：作者脚本中 `root = '.../o_1hours_31day.npy'` 为 **写死** 的绝对路径，需改为你导出的 `o_hours.npy`（或复制并改名）。
- **Python 2 / Spark**：仓库为 **Python 2** 与 **集群 Spark**（`setMaster('yarn')` 等）。在 Windows 本机一般不能原样跑通，需要其一：**Linux + Spark 环境**、**将核心 EM 循环迁到单进程 NumPy**（逻辑已集中在同一文件，可对照论文式 (8)–(11) 核对）、或使用 **WSL / Docker**。
- **超参**：脚本中 `N = 12` 为 **每个 EM 子序列长度**（滑动窗口），`K` 由命令行传入；`X = O[:, :2*24, :]` 表示 **仅用前两日**作训练子集——你可按数据长度改切片。
- **与 `docs/reference.md` 附录一致**：导出 `.npy` 只解决 **观测张量格式**；**共享发射 EM、Viterbi、论文 §5 评测** 仍依赖上述训练/解码代码或自研实现。

### 3. 最小自检

演示数据在导出后形状应类似 **`(n_units, n_hours, L)`**，例如 demo 三天 × 24 小时：`T = 72`。可用：

```python
import numpy as np
O = np.load("data/sshmm/out_demo/o_hours.npy")
print(O.shape, O.min(), O.max())
```

期望 `L` 与 `sshmm_manifest.json` 中 `"L"` 一致，且归一化通道大致落在 \([0,1]\)。

## 与百度手册的差异说明

- **空间单元**：论文为路网分区；项目为 `01_units` 地块。需将 **网格中心 / OD 点** 与地块 **面** 做空间连接（脚本内 `within`）。
- **郊环约束**：若 OD 仅覆盖郊环内 O/D，站域 3 km 必须在产品空间范围内，否则需换全城管网或申请范围。
- **小时覆盖**：若仅部分月份有小时 OD，长序列会断档；`manifest` 中会记录实际时间集合。
