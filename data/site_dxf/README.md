# `site.dxf` 导出与校准

## 目录

- `local/`：从 `site.dxf` 模型空间按图层导出的 GeoJSON，坐标为 **CAD 绘图坐标**（与 `district.dxf` 同一地方坐标系）。
- `crs84/`：经 `data/SITE.json`（WGS84 场地红线）对齐后的 GeoJSON，坐标为 **EPSG:4326**。

## 如何生成

在仓库根目录执行：

```bash
python scripts/align_cad_geojson_to_SITE_json.py
```

该脚本会同时导出 `site.dxf`、`district.dxf`，按 `SITE.json` 拟合 **相似变换 + ICP**（在 EPSG:32651 UTM 平面），并把同一变换应用到 `site_dxf` 与 `district_dxf` 的 `crs84/` 输出。

拟合参数与误差见 `data/cad_alignment/transform_meta.json`。
