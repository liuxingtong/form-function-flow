from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union


CRS_WGS84 = "EPSG:4326"


def _load_site_polygon(site_path: Path) -> gpd.GeoDataFrame:
    site = gpd.read_file(site_path)
    if site.crs is None:
        site = site.set_crs(CRS_WGS84)
    site = site.to_crs(CRS_WGS84)
    geom = site.geometry.iloc[0]
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        geom = Polygon(coords)
    return gpd.GeoDataFrame({"name": ["SITE"]}, geometry=[geom], crs=CRS_WGS84)


def _find_source_files(site_3km: Path) -> tuple[Path | None, Path | None]:
    try:
        building = next(site_3km.glob("01-*/*/*甯﹂珮搴?geojson"))
        landuse = next(site_3km.glob("09-*/*/Data/*寮€婧愬缓璁剧敤鍦?geojson"))
        return building, landuse
    except StopIteration:
        return None, None


def _build_zone_parcels(repo_root: Path) -> gpd.GeoDataFrame:
    district_dir = repo_root / "data" / "district_dxf" / "crs84"
    zone_sources = [
        ("CBD", district_dir / "Z_CBD.geojson"),
        ("TOD", district_dir / "Z_TOD.geojson"),
        ("OFC", district_dir / "Z_OFC.geojson"),
    ]

    rows = []
    for zone_id, fp in zone_sources:
        if not fp.exists():
            continue
        gdf = gpd.read_file(fp)
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_WGS84)
        gdf = gdf.to_crs(CRS_WGS84)

        polygons = list(gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].geometry)
        lines = list(gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].geometry)

        if lines:
            merged = unary_union(lines)
            polygons.extend(list(polygonize(merged)))

        pid = 1
        for geom in polygons:
            if geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                rows.append({"zone_id": zone_id, "parcel_id": f"{zone_id}_{pid}", "geometry": geom})
                pid += 1
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    if not poly.is_empty:
                        rows.append({"zone_id": zone_id, "parcel_id": f"{zone_id}_{pid}", "geometry": poly})
                        pid += 1

    if not rows:
        return gpd.GeoDataFrame({"zone_id": [], "parcel_id": []}, geometry=[], crs=CRS_WGS84)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84)


def prepare(repo_root: Path) -> dict:
    data_dir = repo_root / "data"
    site_3km = data_dir / "site_3km"
    out_dir = data_dir / "site_design_platform" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    site_poly = _load_site_polygon(data_dir / "SITE.json")
    building_path, landuse_path = _find_source_files(site_3km)

    fallback_buildings = out_dir / "site_buildings.geojson"
    fallback_landuse = out_dir / "site_landuse.geojson"

    if building_path and landuse_path:
        buildings = gpd.read_file(building_path)[["id", "Height", "geometry"]].to_crs(CRS_WGS84)
        landuse = gpd.read_file(landuse_path)[["id", "Class", "class2", "geometry"]].to_crs(CRS_WGS84)
        buildings_in = gpd.clip(buildings, site_poly)
        landuse_in = gpd.clip(landuse, site_poly)
        src_building = str(building_path.relative_to(repo_root))
        src_landuse = str(landuse_path.relative_to(repo_root))
    else:
        buildings_in = gpd.read_file(fallback_buildings).to_crs(CRS_WGS84) if fallback_buildings.exists() else gpd.GeoDataFrame({"id": [], "Height": []}, geometry=[], crs=CRS_WGS84)
        landuse_in = gpd.read_file(fallback_landuse).to_crs(CRS_WGS84) if fallback_landuse.exists() else gpd.GeoDataFrame({"id": [], "Class": [], "class2": []}, geometry=[], crs=CRS_WGS84)
        src_building = "fallback:data/site_design_platform/datasets/site_buildings.geojson"
        src_landuse = "fallback:data/site_design_platform/datasets/site_landuse.geojson"

    zone_parcels = _build_zone_parcels(repo_root)
    if len(zone_parcels):
        zone_parcels = gpd.clip(zone_parcels, site_poly)

    buildings_out = out_dir / "site_buildings.geojson"
    landuse_out = out_dir / "site_landuse.geojson"
    zone_parcels_out = out_dir / "zone_parcels.geojson"
    summary_out = out_dir / "summary.json"

    buildings_in.to_file(buildings_out, driver="GeoJSON")
    landuse_in.to_file(landuse_out, driver="GeoJSON")
    zone_parcels.to_file(zone_parcels_out, driver="GeoJSON")

    height_min = float(buildings_in["Height"].min()) if len(buildings_in) and "Height" in buildings_in.columns else None
    height_max = float(buildings_in["Height"].max()) if len(buildings_in) and "Height" in buildings_in.columns else None
    lu_classes = sorted([str(x) for x in landuse_in["class2"].dropna().unique().tolist()]) if len(landuse_in) and "class2" in landuse_in.columns else []

    summary = {
        "buildings_features": int(len(buildings_in)),
        "landuse_features": int(len(landuse_in)),
        "zone_parcels_features": int(len(zone_parcels)),
        "height_min": height_min,
        "height_max": height_max,
        "landuse_classes": lu_classes,
        "source_building": src_building,
        "source_landuse": src_landuse,
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    summary = prepare(repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
