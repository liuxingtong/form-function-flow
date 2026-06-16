# CAD → WGS84 对齐元数据

`transform_meta.json` 由 `scripts/align_cad_geojson_to_SITE_json.py` 生成，记录：

- 用于拟合的 `site.dxf` / `district.dxf` / `data/SITE.json` 路径
- UTM 工作投影（默认 EPSG:32651）
- 相似变换参数（尺度、旋转、平移，作用于 CAD → UTM）
- ICP 拟合后采样点与 DXF 原始顶点到 `SITE.json` 红线的距离统计（米）

重新生成：见 `data/site_dxf/README.md`。
