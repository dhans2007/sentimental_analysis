/**
 * SentimentScope AI - Enterprise Client Controller
 * Handles live predictions, Chart.js analytics, batch file processing,
 * model comparisons, and theme switching.
 */

// Toast notification helper
function showToast(message, duration = 3000) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// --------------------------------------------------------------------------
// Theme Toggle (Dark / Light Mode)
// --------------------------------------------------------------------------
function initTheme() {
  const savedTheme = localStorage.getItem("sentiment_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcons(savedTheme);

  const toggleBtn = document.getElementById("theme-toggle-btn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("sentiment_theme", next);
      updateThemeIcons(next);
    });
  }
}

function updateThemeIcons(theme) {
  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");
  if (!sunIcon || !moonIcon) return;
  if (theme === "light") {
    sunIcon.classList.add("hidden");
    moonIcon.classList.remove("hidden");
  } else {
    sunIcon.classList.remove("hidden");
    moonIcon.classList.add("hidden");
  }
}

// --------------------------------------------------------------------------
// Single Text Analyzer (Index Page)
// --------------------------------------------------------------------------
function initAnalyzer() {
  const textInput = document.getElementById("text-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-input-btn");
  const charCounter = document.getElementById("char-counter");
  const liveToggle = document.getElementById("live-predict-toggle");
  const resultContainer = document.getElementById("result-container");

  if (!textInput || !analyzeBtn) return;

  let debounceTimer = null;

  function updateCharCount() {
    const val = textInput.value;
    const chars = val.length;
    const words = val.trim() ? val.trim().split(/\s+/).length : 0;
    if (charCounter) charCounter.textContent = `${chars} chars • ${words} words`;
  }

  async function performAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
      if (resultContainer) resultContainer.classList.add("hidden");
      return;
    }

    analyzeBtn.disabled = true;
    const originalText = analyzeBtn.querySelector(".btn-text")?.textContent || "Analyze";
    if (analyzeBtn.querySelector(".btn-text")) analyzeBtn.querySelector(".btn-text").textContent = "Analyzing...";

    try {
      const res = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      if (data.error) {
        showToast(data.error);
        return;
      }

      renderAnalyzerResults(data);
    } catch (e) {
      console.error("Analysis failed:", e);
      showToast("Error analyzing sentiment. Please try again.");
    } finally {
      analyzeBtn.disabled = false;
      if (analyzeBtn.querySelector(".btn-text")) analyzeBtn.querySelector(".btn-text").textContent = originalText;
    }
  }

  function renderAnalyzerResults(data) {
    if (!resultContainer) return;
    resultContainer.classList.remove("hidden");

    // Primary verdict & emotion
    const badge = document.getElementById("result-badge");
    const emotionBadge = document.getElementById("emotion-badge");
    const confVal = document.getElementById("result-confidence");

    if (badge) {
      badge.textContent = data.sentiment;
      badge.className = `badge-lg ${data.sentiment}`;
    }
    if (emotionBadge) emotionBadge.textContent = data.emotion || "Neutral";
    if (confVal) confVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Polarity Intensity Gauge
    const intensityVal = document.getElementById("intensity-value");
    const gaugeFill = document.getElementById("gauge-fill");
    if (intensityVal && gaugeFill) {
      const intensity = data.intensity || 0;
      const pct = Math.min(100, Math.max(0, (intensity + 1.0) * 50));
      gaugeFill.style.width = `${pct}%`;
      gaugeFill.className = `gauge-fill ${data.sentiment}`;
      const sign = intensity > 0 ? "+" : "";
      intensityVal.textContent = `${sign}${intensity.toFixed(2)} (${data.sentiment.toUpperCase()})`;
    }

    // Probability bars
    const probs = data.probabilities || {};
    const posP = ((probs.positive || 0) * 100).toFixed(1);
    const neuP = ((probs.neutral || 0) * 100).toFixed(1);
    const negP = ((probs.negative || 0) * 100).toFixed(1);

    document.getElementById("prob-bar-pos").style.width = `${posP}%`;
    document.getElementById("prob-val-pos").textContent = `${posP}%`;

    document.getElementById("prob-bar-neu").style.width = `${neuP}%`;
    document.getElementById("prob-val-neu").textContent = `${neuP}%`;

    document.getElementById("prob-bar-neg").style.width = `${negP}%`;
    document.getElementById("prob-val-neg").textContent = `${negP}%`;

    // Token Highlights (Word Explainability)
    const tokenBox = document.getElementById("token-highlight-box");
    if (tokenBox && data.tokens) {
      tokenBox.innerHTML = "";
      data.tokens.forEach(tok => {
        const span = document.createElement("span");
        if (tok.type === "pos") span.className = "hl-pos";
        else if (tok.type === "neg") span.className = "hl-neg";
        else if (tok.type === "negator") span.className = "hl-negator";
        span.textContent = tok.word + " ";
        tokenBox.appendChild(span);
      });
    }

    // Aspect Breakdown
    const aspectsList = document.getElementById("aspects-list");
    if (aspectsList) {
      aspectsList.innerHTML = "";
      if (data.aspect_details && data.aspect_details.length > 0) {
        data.aspect_details.forEach(asp => {
          const item = document.createElement("div");
          item.className = "aspect-card-item";
          item.innerHTML = `
            <div class="aspect-name-wrap">
              <span class="aspect-title">${asp.aspect}</span>
            </div>
            <div class="aspect-meta">
              <span class="aspect-badge-mini ${asp.sentiment}">${asp.sentiment} (${(asp.confidence * 100).toFixed(0)}%)</span>
            </div>
          `;
          aspectsList.appendChild(item);
        });
      } else {
        aspectsList.innerHTML = `<span class="card-hint">General statement &bull; No specific product aspects detected.</span>`;
      }
    }

    // Bind feedback buttons
    document.querySelectorAll(".feedback-btn").forEach(btn => {
      btn.onclick = () => {
        const action = btn.getAttribute("data-action");
        fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: data.text,
            predicted: data.sentiment,
            type: action
          })
        });
        showToast(action === "thumbs_up" ? "Thanks! Verified as accurate." : "Thanks! Recorded for model retraining.");
      };
    });
  }

  // Event Listeners
  analyzeBtn.addEventListener("click", performAnalysis);

  textInput.addEventListener("input", () => {
    updateCharCount();
    if (liveToggle && liveToggle.checked) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        if (textInput.value.trim().length > 3) performAnalysis();
      }, 400);
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      textInput.value = "";
      updateCharCount();
      if (resultContainer) resultContainer.classList.add("hidden");
    });
  }

  // Quick prompt chips
  document.querySelectorAll(".preset-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const sample = chip.getAttribute("data-sample");
      if (sample) {
        textInput.value = sample;
        updateCharCount();
        performAnalysis();
      }
    });
  });

  // Initial analyze if text preset
  if (textInput.value.trim()) {
    updateCharCount();
    performAnalysis();
  }
}

// --------------------------------------------------------------------------
// Batch File Processor (Batch Page)
// --------------------------------------------------------------------------
function initBatchProcessor() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const downloadSampleBtn = document.getElementById("download-sample-btn");
  const progressCard = document.getElementById("batch-progress-card");
  const progressBar = document.getElementById("batch-progress-bar");
  const progressPercent = document.getElementById("progress-percentage");
  const resultsSection = document.getElementById("batch-results-section");
  const tableBody = document.getElementById("batch-table-body");
  const exportBtn = document.getElementById("batch-export-csv-btn");
  const filterInput = document.getElementById("batch-filter-input");
  const sentimentFilter = document.getElementById("batch-sentiment-filter");

  if (!dropZone || !fileInput) return;

  let currentBatchData = [];

  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) processUploadedFile(e.target.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) processUploadedFile(e.dataTransfer.files[0]);
  });

  if (downloadSampleBtn) {
    downloadSampleBtn.addEventListener("click", () => {
      const sampleCSV = `text\n"The battery life is remarkable, lasted two days without charging!"\n"Terrible customer service, they ignored my emails."\n"The scheduled system maintenance will occur this Friday at 3 PM UTC."\n"Not bad at all, very snappy interface and great value."\n"App keeps crashing whenever I try to upload photos."`;
      const blob = new Blob([sampleCSV], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sentiment_sample_batch.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  async function processUploadedFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    if (progressCard) progressCard.classList.remove("hidden");
    if (progressBar) progressBar.style.width = "40%";
    if (progressPercent) progressPercent.textContent = "40%";

    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      if (progressBar) progressBar.style.width = "100%";
      if (progressPercent) progressPercent.textContent = "100%";

      if (data.error) {
        showToast(data.error);
        if (progressCard) progressCard.classList.add("hidden");
        return;
      }

      currentBatchData = data.results || [];
      setTimeout(() => {
        if (progressCard) progressCard.classList.add("hidden");
        renderBatchResults(data);
      }, 400);
    } catch (e) {
      console.error(e);
      showToast("Error processing batch file.");
      if (progressCard) progressCard.classList.add("hidden");
    }
  }

  function renderBatchResults(data) {
    if (!resultsSection) return;
    resultsSection.classList.remove("hidden");

    document.getElementById("batch-total-count").textContent = data.total;
    document.getElementById("batch-pos-count").textContent = data.summary.positive;
    document.getElementById("batch-neu-count").textContent = data.summary.neutral;
    document.getElementById("batch-neg-count").textContent = data.summary.negative;

    renderTableRows(currentBatchData);
  }

  function renderTableRows(items) {
    if (!tableBody) return;
    tableBody.innerHTML = "";

    const query = (filterInput?.value || "").toLowerCase();
    const sent = sentimentFilter?.value || "all";

    const filtered = items.filter(item => {
      const matchText = item.text.toLowerCase().includes(query);
      const matchSent = sent === "all" || item.sentiment === sent;
      return matchText && matchSent;
    });

    if (filtered.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-subtle);padding:24px;">No matching rows found.</td></tr>`;
      return;
    }

    filtered.forEach((r, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color:var(--text-subtle);font-weight:600;">${idx + 1}</td>
        <td style="max-width:380px;">${r.text}</td>
        <td><span class="badge-tag ${r.sentiment}">${r.sentiment}</span></td>
        <td><strong>${(r.confidence * 100).toFixed(1)}%</strong></td>
        <td><span style="font-size:12px;color:var(--text-muted);">${r.emotion || "—"}</span></td>
        <td><span style="font-size:12px;color:var(--text-subtle);">${r.aspects.join(", ") || "General"}</span></td>
      `;
      tableBody.appendChild(tr);
    });
  }

  if (filterInput) filterInput.addEventListener("input", () => renderTableRows(currentBatchData));
  if (sentimentFilter) sentimentFilter.addEventListener("change", () => renderTableRows(currentBatchData));

  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!currentBatchData.length) return;
      let csv = "Text,Sentiment,Confidence,Emotion,Aspects\n";
      currentBatchData.forEach(r => {
        csv += `"${r.text.replace(/"/g, '""')}",${r.sentiment},${r.confidence},"${r.emotion}","${r.aspects.join('; ')}"\n`;
      });
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sentiment_batch_enriched_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      showToast("Enriched CSV downloaded successfully!");
    });
  }
}

// --------------------------------------------------------------------------
// AI Smart Auto-Responder & Action Extractor (Assistant Page)
// --------------------------------------------------------------------------
function initAssistantController() {
  const input = document.getElementById("assistant-input");
  const customInstInput = document.getElementById("custom-instruction-input");
  const btn = document.getElementById("generate-reply-btn");
  const charCounter = document.getElementById("assistant-char-counter");
  const resultsContainer = document.getElementById("assistant-results-container");
  const urgencyBadge = document.getElementById("urgency-badge");
  const sentimentBadge = document.getElementById("assistant-sentiment-badge");
  const emotionText = document.getElementById("assistant-emotion-text");
  const actionList = document.getElementById("action-items-list");
  const aspectsPills = document.getElementById("assistant-aspects-pills");
  const replyDisplay = document.getElementById("reply-text-display");
  const replyEnginePill = document.getElementById("reply-engine-pill");
  const aiStatusText = document.getElementById("ai-status-text");
  const copyBtn = document.getElementById("copy-reply-btn");
  const copyBtnLabel = document.getElementById("copy-btn-label");
  const draftTabs = document.querySelectorAll(".draft-tabs .draft-tab, .draft-tab");

  if (!input || !btn) return;

  let currentReplies = {};
  let activeTone = "empathetic";

  function updateCharCount() {
    if (charCounter) charCounter.textContent = `${input.value.length} characters`;
  }

  async function generateResponses() {
    const text = input.value.trim();
    const custom_instruction = customInstInput ? customInstInput.value.trim() : "";
    if (!text) return;

    btn.disabled = true;
    const btnSpan = btn.querySelector("span");
    if (btnSpan) btnSpan.textContent = "Generating AI Response...";

    try {
      const res = await fetch("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, custom_instruction })
      });
      const data = await res.json();

      if (data.error) {
        showToast(data.error);
        return;
      }

      currentReplies = data.replies || {};
      renderAssistantResults(data);
    } catch (e) {
      console.error(e);
      showToast("Error generating response draft.");
    } finally {
      btn.disabled = false;
      if (btnSpan) btnSpan.textContent = "Generate AI Plan & Reply";
    }
  }

  function renderAssistantResults(data) {
    if (!resultsContainer) return;
    resultsContainer.classList.remove("hidden");

    // Urgency & Sentiment
    if (urgencyBadge) {
      urgencyBadge.textContent = data.urgency;
      urgencyBadge.className = `urgency-badge ${data.urgency_code}`;
    }
    if (sentimentBadge) {
      sentimentBadge.textContent = data.sentiment;
      sentimentBadge.className = `badge-tag ${data.sentiment}`;
    }
    if (emotionText) emotionText.textContent = data.emotion;

    // AI Provider Badge
    if (replyEnginePill) {
      if (data.ai_powered) {
        replyEnginePill.textContent = "⚡ Gemini AI";
        replyEnginePill.className = "text-[10px] px-2 py-0.5 rounded-full font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      } else {
        replyEnginePill.textContent = "🤖 Built-in Engine";
        replyEnginePill.className = "text-[10px] px-2 py-0.5 rounded-full font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30";
      }
    }

    if (aiStatusText) {
      if (data.ai_powered) {
        aiStatusText.textContent = "Gemini AI Active";
      } else {
        aiStatusText.textContent = "Rule Engine (Add GEMINI_API_KEY for AI)";
      }
    }

    // Action items
    if (actionList) {
      actionList.innerHTML = "";
      (data.action_items || []).forEach(item => {
        const div = document.createElement("div");
        div.className = "action-item-box";
        div.innerHTML = `<span>${item}</span>`;
        actionList.appendChild(div);
      });
    }

    // Aspects
    if (aspectsPills) {
      aspectsPills.innerHTML = "";
      if (data.aspects && data.aspects.length) {
        data.aspects.forEach(a => {
          const span = document.createElement("span");
          span.className = "aspect-chip-sm";
          span.textContent = a;
          aspectsPills.appendChild(span);
        });
      } else {
        aspectsPills.innerHTML = `<span style="font-size:12px;color:var(--text-subtle);">General customer feedback</span>`;
      }
    }

    // Update active reply textarea
    updateReplyText();
  }

  function updateReplyText() {
    if (replyDisplay && currentReplies[activeTone] !== undefined) {
      replyDisplay.value = currentReplies[activeTone];
    }
  }

  // Allow manual edits to be preserved
  if (replyDisplay) {
    replyDisplay.addEventListener("input", () => {
      currentReplies[activeTone] = replyDisplay.value;
    });
  }

  draftTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      draftTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeTone = tab.getAttribute("data-tone");
      updateReplyText();
    });
  });

  if (copyBtn && replyDisplay) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(replyDisplay.value);
      if (copyBtnLabel) copyBtnLabel.textContent = "Copied! ✓";
      showToast("Reply draft copied to clipboard!");
      setTimeout(() => {
        if (copyBtnLabel) copyBtnLabel.textContent = "Copy Draft";
      }, 2000);
    });
  }

  // Preset ticket chips
  document.querySelectorAll(".preset-chips [data-ticket]").forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.getAttribute("data-ticket");
      updateCharCount();
      generateResponses();
    });
  });

  input.addEventListener("input", updateCharCount);
  btn.addEventListener("click", generateResponses);

  // Initial trigger if input preset
  if (input.value.trim()) {
    updateCharCount();
    generateResponses();
  }
}


// --------------------------------------------------------------------------
// Model Arena / Comparator (Comparator Page)
// --------------------------------------------------------------------------
function initComparator() {
  const input = document.getElementById("compare-text-input");
  const btn = document.getElementById("run-compare-btn");
  const grid = document.getElementById("compare-cards-grid");

  if (!input || !btn || !grid) return;

  async function runComparison() {
    const text = input.value.trim();
    if (!text) return;

    btn.disabled = true;
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      renderComparisonCards(data.models || {});
    } catch (e) {
      console.error(e);
      showToast("Error running model benchmark.");
    } finally {
      btn.disabled = false;
    }
  }

  function renderComparisonCards(models) {
    grid.innerHTML = "";
    Object.entries(models).forEach(([name, res]) => {
      const isFeatured = name.includes("Ensemble");
      const card = document.createElement("div");
      card.className = `model-card ${isFeatured ? "featured" : ""}`;

      let probsHtml = "";
      if (res.probs) {
        probsHtml = Object.entries(res.probs).map(([c, p]) => `
          <div class="prob-row">
            <span class="prob-label">${c}</span>
            <div class="prob-track"><div class="prob-fill bg-${c}" style="width:${(p * 100).toFixed(0)}%;"></div></div>
            <span class="prob-percent">${(p * 100).toFixed(0)}%</span>
          </div>
        `).join("");
      }

      card.innerHTML = `
        <div class="model-card-header">
          <div>
            <div class="model-name">${name}</div>
            ${isFeatured ? '<span class="badge-version" style="margin-left:0;margin-top:4px;display:inline-block;">Current Production Engine</span>' : ''}
          </div>
          <span class="badge-lg ${res.label}">${res.label}</span>
        </div>
        <div class="confidence-val" style="font-size:22px;margin-bottom:12px;">${(res.confidence * 100).toFixed(1)}% <span class="confidence-sub">Confidence</span></div>
        <div class="prob-distribution">${probsHtml}</div>
      `;
      grid.appendChild(card);
    });
  }

  btn.addEventListener("click", runComparison);

  document.querySelectorAll("[data-compare]").forEach(btn => {
    btn.addEventListener("click", () => {
      input.value = btn.getAttribute("data-compare");
      runComparison();
    });
  });

  if (input.value.trim()) runComparison();
}

// --------------------------------------------------------------------------
// Analytics Dashboard (Dashboard Page)
// --------------------------------------------------------------------------
function initDashboard() {
  const distCanvas = document.getElementById("sentimentDistributionChart");
  const timeCanvas = document.getElementById("sentimentTimelineChart");
  const aspectCanvas = document.getElementById("aspectChart");
  const emotionCanvas = document.getElementById("emotionChart");

  if (!distCanvas || typeof Chart === "undefined") return;

  async function loadDashboardData() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();

      // Top KPIs
      const total = data.total || 1;
      const pos = data.counts.positive || 0;
      const neu = data.counts.neutral || 0;
      const neg = data.counts.negative || 0;

      document.getElementById("kpi-total").textContent = data.total;
      document.getElementById("kpi-pos-percent").textContent = `${((pos / total) * 100).toFixed(1)}% (${pos})`;
      document.getElementById("kpi-neu-percent").textContent = `${((neu / total) * 100).toFixed(1)}% (${neu})`;
      document.getElementById("kpi-neg-percent").textContent = `${((neg / total) * 100).toFixed(1)}% (${neg})`;

      // 1. Distribution Doughnut Chart
      new Chart(distCanvas, {
        type: "doughnut",
        data: {
          labels: ["Positive", "Neutral", "Negative"],
          datasets: [{
            data: [pos, neu, neg],
            backgroundColor: ["#10b981", "#f59e0b", "#f43f5e"],
            borderWidth: 0,
            hoverOffset: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "bottom", labels: { color: "#9ca3af", font: { family: "Outfit" } } } }
        }
      });

      // 2. Timeline Line Chart
      if (timeCanvas && data.timeline && data.timeline.length) {
        const labels = data.timeline.map(t => t.date);
        new Chart(timeCanvas, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              { label: "Positive", data: data.timeline.map(t => t.positive), borderColor: "#10b981", tension: 0.3, fill: false },
              { label: "Neutral", data: data.timeline.map(t => t.neutral), borderColor: "#f59e0b", tension: 0.3, fill: false },
              { label: "Negative", data: data.timeline.map(t => t.negative), borderColor: "#f43f5e", tension: 0.3, fill: false }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#9ca3af" } } },
            scales: {
              x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#6b7280" } },
              y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#6b7280", stepSize: 1 } }
            }
          }
        });
      }

      // 3. Aspect Bar Chart
      if (aspectCanvas && data.aspects) {
        const aspectLabels = Object.keys(data.aspects);
        const aspectValues = Object.values(data.aspects);
        new Chart(aspectCanvas, {
          type: "bar",
          data: {
            labels: aspectLabels,
            datasets: [{
              label: "Mentions",
              data: aspectValues,
              backgroundColor: "#6366f1",
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { display: false }, ticks: { color: "#9ca3af" } },
              y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#6b7280", stepSize: 1 } }
            }
          }
        });
      }

      // 4. Emotion Doughnut / Polar Chart
      if (emotionCanvas && data.emotions) {
        new Chart(emotionCanvas, {
          type: "polarArea",
          data: {
            labels: Object.keys(data.emotions),
            datasets: [{
              data: Object.values(data.emotions),
              backgroundColor: ["rgba(99,102,241,0.7)", "rgba(16,185,129,0.7)", "rgba(244,63,94,0.7)", "rgba(245,158,11,0.7)", "rgba(6,182,212,0.7)"],
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "right", labels: { color: "#9ca3af", boxWidth: 12 } } },
            scales: { r: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { display: false } } }
          }
        });
      }
    } catch (e) {
      console.error("Dashboard failed:", e);
    }
  }

  loadDashboardData();
}

// --------------------------------------------------------------------------
// History Page Controller (History Page)
// --------------------------------------------------------------------------
function initHistory() {
  const searchInput = document.getElementById("history-search-input");
  const sentimentFilter = document.getElementById("history-sentiment-filter");
  const clearBtn = document.getElementById("clear-history-btn");
  const container = document.getElementById("history-items-container");

  if (!container) return;

  function filterItems() {
    const q = (searchInput?.value || "").toLowerCase();
    const sent = sentimentFilter?.value || "all";

    document.querySelectorAll(".history-item-card").forEach(card => {
      const cardText = card.getAttribute("data-text") || "";
      const cardSent = card.getAttribute("data-sentiment") || "";
      const matchText = cardText.includes(q);
      const matchSent = sent === "all" || cardSent === sent;
      card.style.display = matchText && matchSent ? "block" : "none";
    });
  }

  if (searchInput) searchInput.addEventListener("input", filterItems);
  if (sentimentFilter) sentimentFilter.addEventListener("change", filterItems);

  // Active learning feedback buttons in history
  document.querySelectorAll(".feedback-icon-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const text = btn.getAttribute("data-text");
      const predicted = btn.getAttribute("data-pred");
      const target = btn.getAttribute("data-target");

      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          predicted,
          corrected: target,
          type: "correction"
        })
      });

      showToast(`Classified as ${target.toUpperCase()} & submitted to training pool!`);
      btn.style.borderColor = "var(--primary)";
      btn.style.color = "var(--primary)";
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (confirm("Are you sure you want to delete all stored history?")) {
        await fetch("/api/history/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "clear_all" })
        });
        container.innerHTML = `
          <div class="card glass-card empty-history-card">
            <div class="empty-icon">📂</div>
            <h3>History cleared</h3>
            <p>All recorded analyses have been removed.</p>
          </div>
        `;
        showToast("History cleared.");
      }
    });
  }
}

// --------------------------------------------------------------------------
// URL & Web Content Analyzer (URL Scraper Page)
// --------------------------------------------------------------------------
function initUrlAnalyzer() {
  const input = document.getElementById("target-url-input");
  const btn = document.getElementById("scrape-url-btn");
  const loadingCard = document.getElementById("url-loading-card");
  const resultsContainer = document.getElementById("url-results-container");
  const presetChips = document.querySelectorAll(".preset-chips [data-url]");
  const filterSelect = document.getElementById("sentence-filter-select");
  const heatmapContainer = document.getElementById("sentence-heatmap-list");

  if (!input || !btn) return;

  let currentSentences = [];

  async function analyzeUrl(url) {
    const targetUrl = url || input.value.trim();
    if (!targetUrl) return;

    btn.disabled = true;
    if (loadingCard) loadingCard.classList.remove("hidden");
    if (resultsContainer) resultsContainer.classList.add("hidden");

    try {
      const res = await fetch("/api/scrape-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: targetUrl })
      });
      const data = await res.json();

      if (data.error) {
        showToast(data.error, 4000);
        return;
      }

      currentSentences = data.sentences || [];
      renderUrlResults(data);
    } catch (e) {
      console.error(e);
      showToast("Error scraping or analyzing webpage.");
    } finally {
      btn.disabled = false;
      if (loadingCard) loadingCard.classList.add("hidden");
    }
  }

  function renderUrlResults(data) {
    if (!resultsContainer) return;
    resultsContainer.classList.remove("hidden");

    // Article Meta
    document.getElementById("article-domain").textContent = data.domain;
    document.getElementById("article-title").textContent = data.title;
    document.getElementById("article-words").textContent = `${data.total_words} words`;
    document.getElementById("article-readtime").textContent = `${data.read_time_min} min read`;
    document.getElementById("article-sentences").textContent = `${data.total_sentences} sentences`;

    // 4 KPIs
    const toneEl = document.getElementById("url-overall-tone");
    toneEl.textContent = data.overall_sentiment.toUpperCase();
    toneEl.className = `kpi-value text-${data.overall_sentiment.substring(0, 3)}`;

    const nss = data.net_sentiment_score;
    const sign = nss > 0 ? "+" : "";
    document.getElementById("url-nss-score").textContent = `${sign}${nss.toFixed(1)}`;

    const b = data.sentiment_breakdown;
    document.getElementById("url-ratio-summary").textContent = `${b.positive_pct}% Pos / ${b.negative_pct}% Neg`;

    const dominantEmotion = data.top_positive_quotes.length > data.top_negative_quotes.length ? "Positive & Engaging" : "Critical & Urgent";
    document.getElementById("url-dominant-emotion").textContent = dominantEmotion;

    // Executive Summary Bullets
    const bulletsList = document.getElementById("url-summary-bullets");
    if (bulletsList && data.summary_bullets) {
      bulletsList.innerHTML = "";
      data.summary_bullets.forEach(bullet => {
        const li = document.createElement("li");
        li.textContent = bullet;
        bulletsList.appendChild(li);
      });
    }

    // Aspect Topics
    const aspectsGrid = document.getElementById("url-aspects-list");
    if (aspectsGrid && data.aspect_distribution) {
      aspectsGrid.innerHTML = "";
      Object.entries(data.aspect_distribution).forEach(([asp, count]) => {
        const div = document.createElement("div");
        div.className = "aspect-card-item";
        div.innerHTML = `
          <span class="aspect-title">${asp}</span>
          <span class="aspect-chip-sm" style="font-weight:700;">${count} mentions</span>
        `;
        aspectsGrid.appendChild(div);
      });
    }

    // Sentence Heatmap
    renderSentenceHeatmap();
  }

  function renderSentenceHeatmap() {
    if (!heatmapContainer) return;
    heatmapContainer.innerHTML = "";

    const filterVal = filterSelect ? filterSelect.value : "all";
    const filtered = currentSentences.filter(s => filterVal === "all" || s.sentiment === filterVal);

    if (!filtered.length) {
      heatmapContainer.innerHTML = `<span class="card-hint" style="padding:16px;text-align:center;display:block;">No sentences match this filter.</span>`;
      return;
    }

    filtered.forEach(s => {
      const card = document.createElement("div");
      card.className = `sentence-card sentiment-border-${s.sentiment}`;
      card.innerHTML = `
        <div class="sentence-text">"${s.text}"</div>
        <div class="sentence-meta">
          <div>
            <span class="badge-tag ${s.sentiment}">${s.sentiment}</span>
            <span style="margin-left:6px;color:var(--text-subtle);">Intensity: ${s.intensity > 0 ? '+' : ''}${s.intensity.toFixed(2)}</span>
          </div>
          <span style="color:var(--text-subtle);">${s.aspects.length ? 'Aspects: ' + s.aspects.join(', ') : ''}</span>
        </div>
      `;
      heatmapContainer.appendChild(card);
    });
  }

  if (filterSelect) filterSelect.addEventListener("change", renderSentenceHeatmap);

  btn.addEventListener("click", () => analyzeUrl());

  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const url = chip.getAttribute("data-url");
      input.value = url;
      analyzeUrl(url);
    });
  });

  // Trigger initial preset if input has value
  if (input.value.trim()) {
    analyzeUrl(input.value.trim());
  }
}

// --------------------------------------------------------------------------
// Master Initialization
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initAnalyzer();
  initBatchProcessor();
  initAssistantController();
  initUrlAnalyzer();
  initComparator();
  initDashboard();
  initHistory();
});
