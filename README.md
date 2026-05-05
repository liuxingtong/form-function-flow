# 形 · 功 · 流（上海火车站 3 km）

面向**上海火车站周边约 3 km**研究范围的城市状态识别与推演数据与工具：以 **GMM** 统一「形态—功能—流动」分层状态及综合推演（四人分工流程见主文档）。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `docs/` | 方法、分工与执行说明（主文档：`docs/形+功+流状态场.md`） |
| `data/site_3km/` | 样例底板与专题数据（单元、边表、时间切片、质检与衍生 JSON 等） |
| `scripts/` | 裁切重组、单元/边构建、POI 分时合成、质检出图等 Python 脚本 |

## 环境

Python 3.10+ 建议。脚本侧常见依赖包括：`geopandas`、`pandas`、`numpy`、`shapely`、`pyproj`；出图类脚本另需 `matplotlib`，部分脚本需 `scipy`、`contextily`。可用虚拟环境后按需 `pip install`。

## 运行方式

在仓库根目录执行，例如：

```bash
python scripts/build_site_units_and_edges.py
python scripts/visualize_units_edges_qa.py
```

各脚本文件顶部 docstring 中有参数与输入输出说明。

## 许可与数据

若 `data/` 中含第三方或受限原始数据，转载与公开前请自行核对许可与脱敏要求。
