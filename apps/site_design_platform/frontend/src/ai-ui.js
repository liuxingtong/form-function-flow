export function createAiUI(handlers) {
  const aiBtn = document.getElementById("btn-ai-allocate");
  if (aiBtn) {
    aiBtn.addEventListener("click", () => {
      const prompt = document.getElementById("ai-vision-input")?.value || "";
      if (handlers.onAiAllocate) handlers.onAiAllocate(prompt);
    });
  }

  const audienceSelect = document.getElementById("ai-audience-select");
  const audienceOther = document.getElementById("ai-audience-other");
  if (audienceSelect && audienceOther) {
    const syncAudienceOther = () => {
      audienceOther.style.display = audienceSelect.value === "其它" ? "block" : "none";
    };
    audienceSelect.addEventListener("change", syncAudienceOther);
    syncAudienceOther();
  }

  return {
    renderAiSummary(text) {
      const el = document.getElementById("ai-allocation-summary");
      if (el) el.textContent = text || "";
    },
    renderAiDetails(text) {
      const el = document.getElementById("ai-allocation-details");
      if (el) el.textContent = text || "";
    },
  };
}
