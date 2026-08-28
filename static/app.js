// Global State & API Configuration
const API_BASE = (window.location.protocol === "file:" || (window.location.port !== "8000" && !window.location.hostname.includes("127.0.0.1")))
  ? "http://127.0.0.1:8000"
  : "";

let predictProbChart = null;
let smoteChart = null;
let priceDistChart = null;
let trainingHistoryChart = null;

let currentHmmData = null;
let viterbiObsSequence = ["High Volatility", "Low Volatility", "High Volatility"];

// Sample Financial Headline Presets
const PRESETS = [
  {
    text: "Amcorp Properties jumps 10.87% on solid 3Q earnings growth, record revenue, and dividend announcement.",
    company: "Amcorp Properties"
  },
  {
    text: "Major corporate scandal and fraud allegations trigger massive 18% selloff as regulatory probe deepens.",
    company: "Subur Tiasa"
  },
  {
    text: "Board of Directors concludes annual general meeting with steady operational updates and unchanged targets.",
    company: "Axiata"
  },
  {
    text: "Company announces billion-dollar strategic acquisition of tech firm to expand into high-growth AI markets.",
    company: "Top Glove"
  }
];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  initPredictProbChart();
  
  // Set default preset in predictor
  setHeadlinePreset(0);
  
  // Check backend connectivity and load data
  checkBackendHealth();
  loadHmmData();
  loadDatasetStats();
  loadDatasetSample();
  loadModelMetrics();
  fetchLiveTicker("AAPL");
});

// Check if backend API server is accessible
async function checkBackendHealth() {
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  
  try {
    const res = await fetch(`${API_BASE}/api/metrics`, { method: "GET" });
    if (res.ok) {
      if (statusPill && statusText) {
        statusPill.className = "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
        statusPill.querySelector("span").className = "w-2 h-2 rounded-full bg-emerald-400 animate-pulse";
        statusText.innerText = "LSTM Model Active";
      }
      hideBackendAlert();
    } else {
      throw new Error("Backend response error");
    }
  } catch (err) {
    if (statusPill && statusText) {
      statusPill.className = "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20";
      statusPill.querySelector("span").className = "w-2 h-2 rounded-full bg-rose-400";
      statusText.innerText = "Backend Offline";
    }
    showBackendAlert();
  }
}

function showBackendAlert() {
  let alertEl = document.getElementById("backend-offline-banner");
  if (!alertEl) {
    alertEl = document.createElement("div");
    alertEl.id = "backend-offline-banner";
    alertEl.className = "bg-rose-950/80 border-b border-rose-800 text-rose-200 px-6 py-3 text-xs flex flex-wrap items-center justify-between gap-3 sticky top-[60px] z-40 backdrop-blur-md";
    alertEl.innerHTML = `
      <div class="flex items-center gap-2 font-medium">
        <i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400"></i>
        <span><strong>Backend Server Offline:</strong> Cannot connect to FastAPI at <code class="bg-slate-900 px-1.5 py-0.5 rounded text-rose-300 font-mono">http://127.0.0.1:8000</code>.</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-slate-300">Run in terminal: <code class="bg-slate-900 px-2 py-0.5 rounded text-cyan-300 font-mono">python app.py</code></span>
        <button onclick="checkBackendHealth(); loadHmmData(); loadDatasetStats();" class="px-2.5 py-1 bg-rose-600 hover:bg-rose-500 text-white rounded font-bold transition">Retry</button>
      </div>
    `;
    document.body.insertBefore(alertEl, document.querySelector("main"));
    lucide.createIcons();
  }
}

function hideBackendAlert() {
  const alertEl = document.getElementById("backend-offline-banner");
  if (alertEl) alertEl.remove();
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));

  const targetSection = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetSection) targetSection.classList.remove("hidden");
  if (targetBtn) targetBtn.classList.add("active");

  lucide.createIcons();

  // Resize charts when tab becomes visible
  setTimeout(() => {
    if (tabId === "predict" && predictProbChart) predictProbChart.resize();
    if (tabId === "dataset") {
      if (smoteChart) smoteChart.resize();
      if (priceDistChart) priceDistChart.resize();
    }
    if (tabId === "metrics" && trainingHistoryChart) trainingHistoryChart.resize();
  }, 100);
}

// -----------------------------------------------------------------------------
// TAB 1: LIVE PREDICTION
// -----------------------------------------------------------------------------
function setHeadlinePreset(idx) {
  const item = PRESETS[idx];
  if (!item) return;
  document.getElementById("headline-input").value = item.text;
  document.getElementById("company-input").value = item.company;
}

function clearPredictInput() {
  document.getElementById("headline-input").value = "";
  document.getElementById("company-input").value = "";
  document.getElementById("cleaned-text-preview").innerText = "Awaiting input...";
  document.getElementById("tokens-container").innerHTML = '<span class="text-slate-500 italic">No tokens parsed yet.</span>';
}

function initPredictProbChart() {
  const ctx = document.getElementById("predictProbChart");
  if (!ctx) return;
  
  predictProbChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Downtrend (< -10%)", "Flat (-10% to +10%)", "Uptrend (> +10%)"],
      datasets: [{
        label: "Probability",
        data: [0.0, 0.0, 0.0],
        backgroundColor: [
          "rgba(244, 63, 94, 0.7)",
          "rgba(245, 158, 11, 0.7)",
          "rgba(16, 185, 129, 0.7)"
        ],
        borderColor: [
          "#f43f5e",
          "#f59e0b",
          "#10b981"
        ],
        borderWidth: 1.5,
        borderRadius: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          ticks: {
            color: "#94a3b8",
            callback: (val) => `${Math.round(val * 100)}%`
          },
          grid: { color: "rgba(255, 255, 255, 0.06)" }
        },
        x: {
          ticks: { color: "#cbd5e1", font: { weight: "600" } },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(2)}%`
          }
        }
      }
    }
  });
}

async function runPrediction() {
  const text = document.getElementById("headline-input").value.trim();
  const company = document.getElementById("company-input").value.trim();
  const btn = document.getElementById("predict-btn");

  if (!text) {
    alert("Please enter a news headline or tweet to analyze.");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i> Analyzing...`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, company })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed.");
    }

    const data = await res.json();
    renderPredictionResult(data);
    hideBackendAlert();
  } catch (error) {
    showBackendAlert();
    alert("Prediction Error: " + error.message + "\nMake sure the backend is running with 'python app.py'.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4"></i><span>Analyze & Predict Trend</span>`;
    lucide.createIcons();
  }
}

function renderPredictionResult(data) {
  // Update NLP preprocessing view
  document.getElementById("cleaned-text-preview").innerText = data.cleaned_text || "(empty)";

  const tokensCont = document.getElementById("tokens-container");
  tokensCont.innerHTML = "";
  if (data.tokens && data.tokens.length > 0) {
    data.tokens.forEach(tok => {
      const badge = document.createElement("span");
      badge.className = `token-chip px-2 py-0.5 rounded-md text-[11px] font-mono border ${
        tok.in_vocab
          ? "bg-indigo-500/20 text-indigo-200 border-indigo-500/30"
          : "bg-amber-500/20 text-amber-200 border-amber-500/30"
      }`;
      badge.title = tok.in_vocab ? "Present in GloVe Vocab" : "Out of Vocab (Random Seeded)";
      badge.innerText = tok.token;
      tokensCont.appendChild(badge);
    });
  } else {
    tokensCont.innerHTML = '<span class="text-slate-500 italic">No tokens found.</span>';
  }

  // Update Result Hero
  const trendLabel = document.getElementById("pred-trend-label");
  const summaryEl = document.getElementById("pred-summary");
  const signalText = document.getElementById("signal-text");
  const signalBadge = document.getElementById("signal-badge");
  const confBadge = document.getElementById("pred-confidence-badge");

  trendLabel.innerText = data.prediction;
  summaryEl.innerText = data.sentiment_summary;
  signalText.innerText = data.signal;
  confBadge.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Colors
  if (data.prediction === "uptrend") {
    trendLabel.className = "text-3xl font-extrabold text-emerald-400 tracking-tight mt-1 capitalize";
    signalBadge.className = "px-4 py-3 rounded-2xl bg-emerald-500/20 border border-emerald-500/40 text-center min-w-[130px] self-start sm:self-center shadow-lg shadow-emerald-500/20";
    signalText.className = "text-base font-black text-emerald-300";
  } else if (data.prediction === "downtrend") {
    trendLabel.className = "text-3xl font-extrabold text-rose-400 tracking-tight mt-1 capitalize";
    signalBadge.className = "px-4 py-3 rounded-2xl bg-rose-500/20 border border-rose-500/40 text-center min-w-[130px] self-start sm:self-center shadow-lg shadow-rose-500/20";
    signalText.className = "text-base font-black text-rose-300";
  } else {
    trendLabel.className = "text-3xl font-extrabold text-amber-400 tracking-tight mt-1 capitalize";
    signalBadge.className = "px-4 py-3 rounded-2xl bg-amber-500/20 border border-amber-500/40 text-center min-w-[130px] self-start sm:self-center shadow-lg shadow-amber-500/20";
    signalText.className = "text-base font-black text-amber-300";
  }

  // Probability Bars
  const pUp = (data.probabilities.uptrend * 100).toFixed(1);
  const pFlat = (data.probabilities.flat * 100).toFixed(1);
  const pDown = (data.probabilities.downtrend * 100).toFixed(1);

  document.getElementById("prob-val-uptrend").innerText = `${pUp}%`;
  document.getElementById("prob-val-flat").innerText = `${pFlat}%`;
  document.getElementById("prob-val-downtrend").innerText = `${pDown}%`;

  document.getElementById("prob-bar-uptrend").style.width = `${pUp}%`;
  document.getElementById("prob-bar-flat").style.width = `${pFlat}%`;
  document.getElementById("prob-bar-downtrend").style.width = `${pDown}%`;

  // Chart Update
  if (predictProbChart) {
    predictProbChart.data.datasets[0].data = [
      data.probabilities.downtrend,
      data.probabilities.flat,
      data.probabilities.uptrend
    ];
    predictProbChart.update();
  }
}

// -----------------------------------------------------------------------------
// TAB 2: HIDDEN MARKOV MODEL (HMM)
// -----------------------------------------------------------------------------
async function loadHmmData() {
  const obsType = document.getElementById("hmm-obs-type")?.value || "volatility";
  try {
    const res = await fetch(`${API_BASE}/api/hmm?observation_type=${obsType}`);
    if (!res.ok) throw new Error("Failed to load HMM parameters.");
    currentHmmData = await res.json();
    renderHmmTables(currentHmmData);
    renderViterbiAvailableButtons(currentHmmData.observations);
  } catch (err) {
    console.error("HMM Load Error:", err);
  }
}

function renderHmmTables(data) {
  // 1. Render Pi
  const piCont = document.getElementById("pi-container");
  if (piCont) {
    piCont.innerHTML = `
      <div class="p-3 bg-slate-950 rounded-xl border border-rose-500/30">
        <span class="text-[11px] text-rose-300 font-semibold block">Downtrend</span>
        <span class="text-lg font-bold font-mono text-white mt-1 block">${(data.start_probabilities.downtrend * 100).toFixed(2)}%</span>
      </div>
      <div class="p-3 bg-slate-950 rounded-xl border border-amber-500/30">
        <span class="text-[11px] text-amber-300 font-semibold block">Flat</span>
        <span class="text-lg font-bold font-mono text-white mt-1 block">${(data.start_probabilities.flat * 100).toFixed(2)}%</span>
      </div>
      <div class="p-3 bg-slate-950 rounded-xl border border-emerald-500/30">
        <span class="text-[11px] text-emerald-300 font-semibold block">Uptrend</span>
        <span class="text-lg font-bold font-mono text-white mt-1 block">${(data.start_probabilities.uptrend * 100).toFixed(2)}%</span>
      </div>
    `;
  }

  // 2. Render Transition Matrix (A)
  const transTable = document.getElementById("trans-matrix-table");
  if (transTable) {
    let headerHtml = `<thead><tr><th class="p-2.5 text-left">From \\ To</th>`;
    data.states.forEach(s => {
      headerHtml += `<th class="p-2.5 capitalize">${s}</th>`;
    });
    headerHtml += `</tr></thead><tbody class="divide-y divide-slate-800 font-mono">`;

    let bodyHtml = "";
    data.states.forEach(fromState => {
      bodyHtml += `<tr><td class="p-2.5 font-sans font-semibold capitalize text-slate-300 text-left bg-slate-900/50">${fromState}</td>`;
      data.states.forEach(toState => {
        const val = data.transition_matrix[fromState]?.[toState] ?? 0;
        const pct = (val * 100).toFixed(2);
        const bgAlpha = Math.min(val * 0.8, 0.4);
        bodyHtml += `<td class="p-2.5" style="background-color: rgba(99, 102, 241, ${bgAlpha})">${pct}%</td>`;
      });
      bodyHtml += `</tr>`;
    });
    bodyHtml += `</tbody>`;
    transTable.innerHTML = headerHtml + bodyHtml;
  }

  // 3. Render Emission Matrix (B)
  const emitTable = document.getElementById("emission-matrix-table");
  if (emitTable) {
    let headerHtml = `<thead><tr><th class="p-2.5 text-left">State \\ Obs</th>`;
    data.observations.forEach(o => {
      headerHtml += `<th class="p-2.5 whitespace-nowrap">${o}</th>`;
    });
    headerHtml += `</tr></thead><tbody class="divide-y divide-slate-800 font-mono">`;

    let bodyHtml = "";
    data.states.forEach(state => {
      bodyHtml += `<tr><td class="p-2.5 font-sans font-semibold capitalize text-slate-300 text-left bg-slate-900/50">${state}</td>`;
      data.observations.forEach(obs => {
        const val = data.emission_matrix[state]?.[obs] ?? 0;
        const pct = (val * 100).toFixed(2);
        const bgAlpha = Math.min(val * 0.8, 0.4);
        bodyHtml += `<td class="p-2.5" style="background-color: rgba(168, 85, 247, ${bgAlpha})">${pct}%</td>`;
      });
      bodyHtml += `</tr>`;
    });
    bodyHtml += `</tbody>`;
    emitTable.innerHTML = headerHtml + bodyHtml;
  }
}

function renderViterbiAvailableButtons(observations) {
  const container = document.getElementById("obs-available-buttons");
  if (!container) return;
  container.innerHTML = "";
  observations.forEach(obs => {
    const btn = document.createElement("button");
    btn.className = "px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-purple-600/30 text-purple-200 border border-slate-700 hover:border-purple-500 transition";
    btn.innerText = `+ ${obs}`;
    btn.onclick = () => addViterbiObservation(obs);
    container.appendChild(btn);
  });
  renderViterbiSequenceUI();
}

function addViterbiObservation(obs) {
  viterbiObsSequence.push(obs);
  renderViterbiSequenceUI();
}

function removeViterbiObservation(idx) {
  viterbiObsSequence.splice(idx, 1);
  renderViterbiSequenceUI();
}

function clearViterbi() {
  viterbiObsSequence = [];
  renderViterbiSequenceUI();
  document.getElementById("viterbi-result-container").classList.add("hidden");
}

function renderViterbiSequenceUI() {
  const container = document.getElementById("viterbi-builder");
  if (!container) return;

  if (viterbiObsSequence.length === 0) {
    container.innerHTML = `<span class="text-slate-500 italic text-xs">No observations selected. Click the buttons below to build sequence.</span>`;
    return;
  }

  container.innerHTML = "";
  viterbiObsSequence.forEach((obs, idx) => {
    const pill = document.createElement("div");
    pill.className = "flex items-center gap-1.5 px-3 py-1 rounded-lg bg-purple-500/20 text-purple-300 border border-purple-500/30 text-xs font-semibold";
    pill.innerHTML = `
      <span class="text-[10px] text-purple-400 font-mono">t=${idx+1}</span>
      <span>${obs}</span>
      <button onclick="removeViterbiObservation(${idx})" class="hover:text-rose-400 ml-1">&times;</button>
    `;
    container.appendChild(pill);
  });
}

async function runViterbi() {
  if (viterbiObsSequence.length === 0) {
    alert("Please add at least one observation to the sequence.");
    return;
  }

  const obsType = document.getElementById("hmm-obs-type")?.value || "volatility";

  try {
    const res = await fetch(`${API_BASE}/api/hmm/viterbi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        observations: viterbiObsSequence,
        observation_type: obsType
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Viterbi decoding failed.");
    }

    const result = await res.json();
    renderViterbiResult(result);
  } catch (error) {
    alert("Viterbi Error: " + error.message);
  }
}

function renderViterbiResult(res) {
  const container = document.getElementById("viterbi-result-container");
  const stepsCont = document.getElementById("viterbi-path-steps");
  const probEl = document.getElementById("viterbi-path-prob");

  container.classList.remove("hidden");
  probEl.innerText = `Joint Path Probability: ${res.path_probability.toExponential(4)}`;

  stepsCont.innerHTML = "";
  res.step_details.forEach((step, idx) => {
    const isUptrend = step.assigned_state === "uptrend";
    const isDowntrend = step.assigned_state === "downtrend";
    const colorClass = isUptrend
      ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-300"
      : isDowntrend
      ? "border-rose-500/40 bg-rose-950/30 text-rose-300"
      : "border-amber-500/40 bg-amber-950/30 text-amber-300";

    const card = document.createElement("div");
    card.className = `p-3 rounded-xl border ${colorClass} min-w-[130px] text-center shadow-lg`;
    card.innerHTML = `
      <span class="text-[10px] uppercase font-bold text-slate-400 block">Step ${step.step}</span>
      <span class="text-xs font-mono text-slate-300 block mb-1">Obs: ${step.observation}</span>
      <span class="text-sm font-extrabold capitalize block">${step.assigned_state}</span>
    `;

    stepsCont.appendChild(card);

    if (idx < res.step_details.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "text-slate-600 font-bold text-sm self-center";
      arrow.innerHTML = "&rarr;";
      stepsCont.appendChild(arrow);
    }
  });
}

// -----------------------------------------------------------------------------
// TAB 3: DATASET & EDA (SMOTE)
// -----------------------------------------------------------------------------
async function loadDatasetStats() {
  try {
    const res = await fetch(`${API_BASE}/api/dataset-stats`);
    if (!res.ok) return;
    const data = await res.json();

    initSmoteChart(data.class_distributions);
    initPriceDistChart(data.price_histogram);
  } catch (err) {
    console.error("Dataset stats load error:", err);
  }
}

function initSmoteChart(distributions) {
  const ctx = document.getElementById("smoteComparisonChart");
  if (!ctx || !distributions) return;

  const raw = distributions.before_smote || { downtrend: 365, flat: 15466, uptrend: 854 };
  const smote = distributions.after_smote || { downtrend: 15466, flat: 15466, uptrend: 15466 };

  if (smoteChart) smoteChart.destroy();

  smoteChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Downtrend", "Flat", "Uptrend"],
      datasets: [
        {
          label: "Original Class Count",
          data: [raw.downtrend || 0, raw.flat || 0, raw.uptrend || 0],
          backgroundColor: "rgba(99, 102, 241, 0.8)",
          borderColor: "#6366f1",
          borderRadius: 6
        },
        {
          label: "After SMOTE Resampling",
          data: [smote.downtrend || 0, smote.flat || 0, smote.uptrend || 0],
          backgroundColor: "rgba(16, 185, 129, 0.8)",
          borderColor: "#10b981",
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.06)" }
        },
        x: {
          ticks: { color: "#cbd5e1" },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { labels: { color: "#cbd5e1" } }
      }
    }
  });
}

function initPriceDistChart(hist) {
  const ctx = document.getElementById("priceDistChart");
  if (!ctx || !hist || !hist.bins) return;

  if (priceDistChart) priceDistChart.destroy();

  priceDistChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: hist.bins.map(b => `${(b * 100).toFixed(0)}%`),
      datasets: [{
        label: "Sample Count",
        data: hist.counts,
        fill: true,
        backgroundColor: "rgba(6, 182, 212, 0.2)",
        borderColor: "#06b6d4",
        borderWidth: 2,
        tension: 0.35,
        pointRadius: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.06)" }
        },
        x: {
          ticks: { color: "#cbd5e1", maxTicksLimit: 12 },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

async function loadDatasetSample() {
  const split = document.getElementById("sample-split-select")?.value || "train";
  const trend = document.getElementById("sample-trend-select")?.value || "all";
  const tbody = document.getElementById("dataset-table-body");

  try {
    const res = await fetch(`${API_BASE}/api/dataset-sample?split=${split}&trend=${trend}&limit=12`);
    if (!res.ok) return;
    const data = await res.json();

    tbody.innerHTML = "";
    data.sample.forEach(row => {
      const relChange = typeof row.Relative_Change === "number" ? `${(row.Relative_Change * 100).toFixed(2)}%` : "N/A";
      const tr = document.createElement("tr");

      const trendBadge = row.Trend === "uptrend"
        ? '<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold text-[11px]">uptrend</span>'
        : row.Trend === "downtrend"
        ? '<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold text-[11px]">downtrend</span>'
        : '<span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold text-[11px]">flat</span>';

      tr.innerHTML = `
        <td class="p-3 text-slate-200 max-w-sm truncate font-sans" title="${row.Title || ''}">${row.Title || ''}</td>
        <td class="p-3 text-slate-400 whitespace-nowrap">${row.Name || 'N/A'}</td>
        <td class="p-3 text-slate-300">${row.Before ?? 'N/A'}</td>
        <td class="p-3 text-slate-300">${row.After ?? 'N/A'}</td>
        <td class="p-3 text-cyan-300 font-bold">${relChange}</td>
        <td class="p-3">${trendBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading dataset sample:", err);
  }
}

// -----------------------------------------------------------------------------
// TAB 4: MODEL ARCHITECTURE & METRICS
// -----------------------------------------------------------------------------
async function loadModelMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) return;
    const metrics = await res.json();

    // KPIs
    document.getElementById("metric-test-acc").innerText = `${(metrics.accuracy * 100).toFixed(2)}%`;
    if (metrics.classification_report?.["macro avg"]) {
      document.getElementById("metric-macro-f1").innerText = metrics.classification_report["macro avg"]["f1-score"].toFixed(3);
    }
    if (metrics.classification_report?.["weighted avg"]) {
      document.getElementById("metric-weighted-f1").innerText = metrics.classification_report["weighted avg"]["f1-score"].toFixed(3);
    }

    initTrainingHistoryChart(metrics.history);
    renderConfusionMatrix(metrics.confusion_matrix);
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

function initTrainingHistoryChart(hist) {
  const ctx = document.getElementById("trainingHistoryChart");
  if (!ctx || !hist) return;

  if (trainingHistoryChart) trainingHistoryChart.destroy();

  trainingHistoryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: hist.epochs.map(e => `Epoch ${e}`),
      datasets: [
        {
          label: "Train Accuracy",
          data: hist.train_accuracy,
          borderColor: "#10b981",
          backgroundColor: "#10b981",
          tension: 0.3
        },
        {
          label: "Val Accuracy",
          data: hist.val_accuracy,
          borderColor: "#6366f1",
          backgroundColor: "#6366f1",
          tension: 0.3
        },
        {
          label: "Train Loss",
          data: hist.train_loss,
          borderColor: "#f59e0b",
          backgroundColor: "#f59e0b",
          borderDash: [4, 4],
          tension: 0.3
        },
        {
          label: "Val Loss",
          data: hist.val_loss,
          borderColor: "#f43f5e",
          backgroundColor: "#f43f5e",
          borderDash: [4, 4],
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.06)" }
        },
        x: {
          ticks: { color: "#cbd5e1" },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { labels: { color: "#cbd5e1" } }
      }
    }
  });
}

function renderConfusionMatrix(matrix) {
  const cont = document.getElementById("conf-matrix-grid");
  if (!cont || !matrix) return;

  const labels = ["Downtrend", "Flat", "Uptrend"];
  let html = `<div class="grid grid-cols-4 gap-1 text-center text-xs font-mono">`;

  html += `<div></div>`;
  labels.forEach(l => {
    html += `<div class="p-2 text-slate-400 font-bold font-sans">${l}</div>`;
  });

  labels.forEach((actualLabel, rowIdx) => {
    html += `<div class="p-2 text-slate-400 font-bold font-sans text-right self-center">${actualLabel}</div>`;
    matrix[rowIdx].forEach((val, colIdx) => {
      const isDiag = rowIdx === colIdx;
      const bg = isDiag ? "bg-indigo-600/30 border-indigo-500/50 text-indigo-200" : "bg-slate-900/60 border-slate-800 text-slate-400";
      html += `<div class="p-3 rounded-lg border ${bg} font-bold">${val}</div>`;
    });
  });

  html += `</div>`;
  cont.innerHTML = html;
}

// -----------------------------------------------------------------------------
// TAB 5: LIVE MARKET FEEDS (YFINANCE)
// -----------------------------------------------------------------------------
function quickTicker(sym) {
  document.getElementById("live-symbol-input").value = sym;
  fetchLiveTicker(sym);
}

async function fetchLiveTicker(sym) {
  const symbol = sym || document.getElementById("live-symbol-input")?.value?.trim() || "AAPL";
  const newsList = document.getElementById("live-news-list");
  newsList.innerHTML = `<div class="text-center py-8 text-slate-500 text-xs"><i data-lucide="loader-2" class="w-5 h-5 animate-spin mx-auto mb-2 text-pink-400"></i> Fetching live quotes and evaluating headline sentiment...</div>`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API_BASE}/api/live-ticker?symbol=${encodeURIComponent(symbol)}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to fetch live data.");
    }

    const data = await res.json();
    document.getElementById("live-ticker-symbol").innerText = data.symbol;
    document.getElementById("live-ticker-price").innerText = data.price ? `$${data.price.toFixed(2)}` : "N/A";

    const changeEl = document.getElementById("live-ticker-change");
    const isPositive = data.change_pct >= 0;
    changeEl.innerText = `${isPositive ? '+' : ''}${data.change_pct.toFixed(2)}%`;
    changeEl.className = `text-xs font-semibold font-mono mt-0.5 ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`;

    if (!data.news || data.news.length === 0) {
      newsList.innerHTML = `<div class="text-center py-6 text-slate-500 text-xs">No recent news articles found for this ticker.</div>`;
      return;
    }

    newsList.innerHTML = "";
    data.news.forEach(item => {
      const isUptrend = item.prediction === "uptrend";
      const isDowntrend = item.prediction === "downtrend";
      const badgeColor = isUptrend
        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
        : isDowntrend
        ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
        : "bg-amber-500/20 text-amber-300 border-amber-500/30";

      const card = document.createElement("div");
      card.className = "glass-card p-4 rounded-xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3";
      card.innerHTML = `
        <div class="space-y-1 max-w-2xl">
          <span class="text-[10px] font-semibold text-slate-500 uppercase">${item.publisher}</span>
          <a href="${item.link}" target="_blank" class="block text-xs font-bold text-slate-100 hover:text-indigo-400 transition leading-snug">${item.title}</a>
        </div>
        <div class="flex items-center gap-2 self-start sm:self-center">
          <span class="px-2.5 py-1 rounded-lg text-xs font-bold uppercase border ${badgeColor}">${item.prediction}</span>
          <button onclick="fillAndPredict('${item.title.replace(/'/g, "\\'")}')" class="p-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white transition" title="Load into Predictor">
            <i data-lucide="arrow-up-right" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      `;
      newsList.appendChild(card);
    });

    lucide.createIcons();
  } catch (error) {
    newsList.innerHTML = `<div class="text-center py-6 text-rose-400 text-xs">Error fetching ticker: ${error.message}</div>`;
  }
}

function fillAndPredict(headline) {
  switchTab("predict");
  document.getElementById("headline-input").value = headline;
  runPrediction();
}
