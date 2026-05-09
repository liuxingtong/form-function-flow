# 空表头模板（从 gdb 导出为 CSV 后，将列名改成与之一致或含下列候选名以便脚本自动识别）

## point_flow_hour_template.csv

列含义：时间、中心点 WGS84、该网格人数

```text
time,x,y,count
2021-04-01 08:00:00,121.451,31.249,1200
```

`time` 亦可用 `timestamp` / `时间`；`count` 亦可用 `人数` / `数量`。

## point_OD_hour_O_template.csv

列含义：时间、起点坐标（与 O 表一致即可）、该 OD 对人数

```text
time,ox,oy,count
2021-04-01 08:00:00,121.45,31.25,50
```

若平台列为 `JobX`/`HousingX` 等，可保留原名；脚本会按候选名匹配（见 `scripts/build_sshmm_observations.py` 中 `consume_od_points`）。

## point_OD_hour_D_template.csv

```text
time,dx,dy,count
2021-04-01 08:00:00,121.46,31.25,50
```

## poi_static_units_template.csv

由「每个 unit 的 POI 分业态计数」叠加得到；列名可自定义，须为**数值**列。

```text
unit_id,商业,办公,交通,居住,公服,教育,医疗,绿地,其他
plot_10648,12,3,5,0,1,0,0,0,0
```

与 `01_units` 的 `unit_id` 必须一致。
