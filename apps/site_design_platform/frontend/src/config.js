export const DATASETS = {
  siteBoundary: "/data/site_dxf/crs84/SITE.geojson",
  buildings: "/data/site_design_platform/datasets/site_buildings.geojson",
  landuse: "/data/site_design_platform/datasets/site_landuse.geojson",
  zoneParcels: "/data/site_design_platform/datasets/zone_parcels.geojson",
  rhinoParcels: "/api/site-design/rhino/parcels",
  rhinoOriginalBuildings: "/api/site-design/rhino/original-buildings",
  rhinoWalking: "/api/site-design/rhino/walking",
  rhinoGround: "/api/site-design/rhino/ground",
  summary: "/data/site_design_platform/datasets/summary.json",
  zCBD: "/data/district_dxf/crs84/Z_CBD.geojson",
  zTOD: "/data/district_dxf/crs84/Z_TOD.geojson",
  zOFC: "/data/district_dxf/crs84/Z_OFC.geojson",
  zRES: "/data/district_dxf/crs84/Z_OFC.geojson",
};

export const LANDUSE_COLORS = {
  RESIDENTIAL: "#f4a261",
  COMMERCIAL: "#e76f51",
  OFFICE: "#b56576",
  INDUSTRIAL: "#6d597a",
  ADMIN: "#457b9d",
  MEDICAL: "#2a9d8f",
  TRANSPORT: "#4d4d4d",
  GREEN: "#7cb342",
};

export const DISTRICT_STYLE = {
  zCBD: { color: "#2f6df6", id: "zone-cbd" },
  zTOD: { color: "#17bebb", id: "zone-tod" },
  zOFC: { color: "#f7b32b", id: "zone-ofc" },
  zRES: { color: "#66bb6a", id: "zone-res" },
};

export const FUNCTION_TYPES = [
  "CENTER_OFFICE",
  "SMALL_OFFICE",
  "APARTMENT",
  "HOTEL",
  "RESIDENTIAL",
  "CENTER_COMMERCIAL",
  "LEISURE_COMMERCIAL",
  "PUBLIC",
  "GROUND",
  "GREEN",
  "COVER_MASS",
  "WALKWAY",
  "HIGHWAY",
];

export const FUNCTION_COLORS = {
  CENTER_OFFICE: "#0b3c9d",
  SMALL_OFFICE: "#2f80ed",
  APARTMENT: "#2e7d32",
  HOTEL: "#e53935",
  RESIDENTIAL: "#7cb342",
  CENTER_COMMERCIAL: "#fb8c00",
  LEISURE_COMMERCIAL: "#fdd835",
  PUBLIC: "#00897b",
  GROUND: "#bfc5cc",
  GREEN: "#26a69a",
  COVER_MASS: "#8d6e63",
  WALKWAY: "#9e9e9e",
  HIGHWAY: "#424242",
};
