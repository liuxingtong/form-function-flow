export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export async function loadDatasets(paths) {
  const [siteBoundary, buildings, landuse, zoneParcels, zCBD, zTOD, zOFC, zRES, summary] = await Promise.all([
    loadJSON(paths.siteBoundary),
    loadJSON(paths.buildings),
    loadJSON(paths.landuse),
    loadJSON(paths.zoneParcels),
    loadJSON(paths.zCBD),
    loadJSON(paths.zTOD),
    loadJSON(paths.zOFC),
    loadJSON(paths.zRES),
    loadJSON(paths.summary),
  ]);
  let rhinoParcels = { type: "FeatureCollection", features: [] };
  let rhinoOriginalBuildings = { type: "FeatureCollection", features: [] };
  let rhinoWalking = { type: "FeatureCollection", features: [] };
  let rhinoGround = { type: "FeatureCollection", features: [] };
  try { rhinoParcels = await loadJSON(paths.rhinoParcels); } catch {}
  try { rhinoOriginalBuildings = await loadJSON(paths.rhinoOriginalBuildings); } catch {}
  try { rhinoWalking = await loadJSON(paths.rhinoWalking); } catch {}
  try { rhinoGround = await loadJSON(paths.rhinoGround); } catch {}
  const hasRhinoParcels = Array.isArray(rhinoParcels.features) && rhinoParcels.features.length > 0;
  const splitByLayer = (layerName, fallbackFc) => {
    if (!hasRhinoParcels) return fallbackFc;
    return {
      type: "FeatureCollection",
      features: rhinoParcels.features.filter((f) => String(f?.properties?.layer || f?.properties?.zone_id || "").toUpperCase() === layerName),
    };
  };
  const zCBD2 = splitByLayer("Z_CBD", zCBD);
  const zTOD2 = splitByLayer("Z_TOD", zTOD);
  const zOFC2 = splitByLayer("Z_OFC", zOFC);
  const zRES2 = splitByLayer("Z_RES", zRES);
  const zoneParcels2 = hasRhinoParcels ? rhinoParcels : zoneParcels;
  return { siteBoundary, buildings, landuse, zoneParcels: zoneParcels2, zCBD: zCBD2, zTOD: zTOD2, zOFC: zOFC2, zRES: zRES2, rhinoOriginalBuildings, rhinoWalking, rhinoGround, summary };
}
