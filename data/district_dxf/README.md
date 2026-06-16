# `district.dxf` 按图层 GeoJSON 导出与校准

## 目录

- `local/`：从 `F:\Aworks\2026studio\shanghaistation\district.dxf` 模型空间按 **图层名** 导出的 GeoJSON，坐标为 **CAD 绘图坐标**（非经纬度）。
- `crs84/`：与 `site.dxf` 使用 **同一套** CAD→WGS84 变换（由 `data/SITE.json` 场地红线与 `site.dxf` 的 SITE 边界在 UTM 平面 ICP 拟合得到），便于与 `data/site_dxf/crs84/`、`data/SITE.json` 同屏叠加。

## 元数据与脚本

- 对齐参数与误差：`data/cad_alignment/transform_meta.json`
- 一键导出并对齐：`python scripts/align_cad_geojson_to_SITE_json.py`（见 `data/site_dxf/README.md`）

## 仅导出本地 CAD 坐标（不对齐）

```bash
python scripts/export_dxf_layers_to_geojson.py ^
  --dxf "F:\Aworks\2026studio\shanghaistation\district.dxf" ^
  --out-dir data/district_dxf/local
```

（Linux/macOS 将 `^` 换为 `\`。）
