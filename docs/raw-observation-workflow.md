# 原始观测分析流程

## 原则

当前分析只保留两类输入：

- 真实静态空间数据：地块、道路、建筑、POI、AOI、蓝绿、房价、公服等
- 真实动态观测数据：带原始时间戳的人流、客流、OD、停留、签到或路况记录

不再使用：

- 八时段切片表
- POI 分时段合成权重
- 基于时段权重构造的 flow / stay / congestion proxy
- synthetic flow 的分时段结果作为正式分析依据

上述旧文件均已归档到 `_deprecated/`。

## 保留的数据底座

- 空间单元：`data/site_3km/01_units.gpkg`
- 单元连接：`data/site_3km/02_edges.csv`
- 地铁真实观测：
  - `data/site_3km/metroflow/inout_10min_3km.parquet`
  - `data/site_3km/metroflow/od_internal_10min_3km.parquet`
- 研究区内其他原始数据源：LBS、点评、美团、路况、停车、出租、骑行等，只要保留真实时间戳即可接入

## 推荐处理链

### 1. 统一空间单元

所有数据先对齐到 `unit_id`。

- 点数据：空间连接到 `01_units.gpkg`
- 线数据：按长度占比、最近边或交点规则汇总到 `unit_id`
- OD 数据：分别落到 origin / destination 对应 `unit_id`

### 2. 保留原始时间粒度

不要先压缩成“早高峰/晚高峰/周末夜间”之类的人造时段。

建议直接保留：

- 10 分钟
- 15 分钟
- 30 分钟
- 1 小时

后续统计时再按问题需要聚合，而不是预先写死八段。

### 3. 构造原始观测长表

目标主键：

- `unit_id`
- `timestamp`

最少保留字段建议：

- `arrivals`
- `departures`
- `stays`
- `od_in`
- `od_out`
- `poi_cat_*`
- `is_weekend`
- `hour`
- `date`

如果你用现有脚本，优先从这里开始：

```bash
python scripts/build_sshmm_observations.py --help
```

这个脚本的有效方向是把真实点流/OD/POI 汇总成 `unit_id x timestamp` 观测表，而不是生成八时段代理。

### 4. 做原始时序特征，而不是时段代理

建议直接从真实时间序列提特征，例如：

- 日均到达峰值、离开峰值、停留峰值
- 峰值出现时刻
- 工作日/周末差值
- 昼夜波动幅度
- 波动稳定性
- 连续高压时长
- 净流入率
- 停留转化率
- OD 指向集中度

这些都可以直接由原始 `timestamp` 序列计算，不需要中间代理时段。

### 5. 分析输出的推荐结构

建议把分析结果分成两张表：

1. `parcel_static_metrics.csv`
   - 每个 `unit_id` 一行
   - 放形态、功能底盘、长期均值、稳定性、结构性约束

2. `parcel_temporal_metrics.csv`
   - 每个 `unit_id x timestamp` 或 `unit_id x date`
   - 放真实观测值和短时变化

如果后面要继续做状态模型，也建议基于原始小时级或 10 分钟级观测做，而不是回到八时段模板。

## 清理后的口径

当前仓库默认口径是：

- 正式分析：只用原始观测
- 旧代理：只保留在 `_deprecated/` 供追溯
- 新结果：不得再把八时段代理表写回 `data/` 或主输出目录
