export async function requestClusterGeneration(payload) {
  // Deprecated: this endpoint is planned for removal in a future release.
  const r = await fetch("/api/site-design/cluster/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`cluster generate failed: ${r.status} ${t}`);
  }
  return r.json();
}
