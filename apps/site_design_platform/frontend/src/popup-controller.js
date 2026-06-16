export function bindInfoPopups(map) {
  // Keep only building hover cursor behavior; disable landuse popup interactions.
  ["buildings-extrusion"].forEach((id) => {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  });
}
