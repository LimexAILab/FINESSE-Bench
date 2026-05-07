const CHART_COLORS = [
  '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

const state = {
  leaderboardMetric: 'public',
  compareMetric: 'public',
  leaderboardSearch: '',
  selectedModels: [
    'Claude Sonnet 4.6',
    'Kimi K2.5',
    'GPT-5.2',
    'Qwen3.5-Plus-02-15',
    'GLM-5',
  ],
  showCI: false,
};

let compareChartInstance = null;
let transferGapChartInstance = null;
let difficultyChartInstance = null;

Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;

const errorBarPlugin = {
  id: 'errorBars',
  afterDatasetsDraw(chart, args, pluginOptions) {
    if (!pluginOptions?.enabled || !pluginOptions?.errorBars?.length) return;
    const { ctx, scales } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.25;

    meta.data.forEach((bar, i) => {
      const error = pluginOptions.errorBars[i];
      if (!error) return;
      const x = bar.x;
      const yTop = scales.y.getPixelForValue(error.high);
      const yBottom = scales.y.getPixelForValue(error.low);

      ctx.beginPath();
      ctx.moveTo(x, yTop);
      ctx.lineTo(x, yBottom);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x - 5, yTop);
      ctx.lineTo(x + 5, yTop);
      ctx.moveTo(x - 5, yBottom);
      ctx.lineTo(x + 5, yBottom);
      ctx.stroke();
    });

    ctx.restore();
  },
};

Chart.register(errorBarPlugin);

function metricLabel(key) {
  const found = ALL_METRICS.find(m => m.key === key);
  return found ? found.label : key;
}

function getModel(modelName) {
  return MODEL_DATA.find(m => m.model === modelName);
}

function getMetricValue(modelName, metric) {
  const model = getModel(modelName);
  return model?.metrics?.[metric] ?? null;
}

function getCI(modelName, metric) {
  return CI_DATA?.[modelName]?.[metric] ?? null;
}

function pct(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function delta(a, b) {
  if (a == null || b == null) return null;
  return a - b;
}

function mean(arr) {
  return arr.reduce((s, x) => s + x, 0) / arr.length;
}

function std(arr) {
  const m = mean(arr);
  return Math.sqrt(mean(arr.map(x => (x - m) ** 2)));
}

function topRows(metric, count = 3) {
  return [...MODEL_DATA]
    .filter(row => row.metrics[metric] != null)
    .sort((a, b) => b.metrics[metric] - a.metrics[metric])
    .slice(0, count);
}

function getTop5Overall() {
  return [...MODEL_DATA]
    .map(row => {
      const values = ['public', 'exam', 'trading'].map(k => row.metrics[k]).filter(v => v != null);
      return { model: row.model, score: mean(values) };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(x => x.model);
}

function populateSelect(selectId, value) {
  const select = document.getElementById(selectId);
  select.innerHTML = ALL_METRICS.map(
    m => `<option value="${m.key}" ${m.key === value ? 'selected' : ''}>${m.label}</option>`
  ).join('');
}

function copyBibtex() {
  const code = document.querySelector('.bibtex-block code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    const btn = document.querySelector('.bibtex-block + button, .bibtex-block ~ button');
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

function renderGroupCards() {
  const container = document.getElementById('groupCards');
  container.innerHTML = GROUP_METRICS.map(group => {
    const top = topRows(group.key, 3);
    return `
      <article class="group-card">
        <h3>${group.label}</h3>
        <p class="mt-2">${GROUP_DESCRIPTIONS[group.key]}</p>
        <div class="mt-4">
          ${top.map((row, i) => `
            <div class="mini-rank">
              <span class="${i === 0 ? 'rank-gold' : i === 1 ? 'rank-silver' : 'rank-bronze'}">${i + 1}</span>
              <span>${row.model}</span>
              <span class="mini-rank-score">${pct(row.metrics[group.key])}</span>
            </div>
          `).join('')}
        </div>
        <div class="mt-4">
          <button class="btn-secondary" onclick="jumpToMetric('${group.key}')">View filtered leaderboard</button>
        </div>
      </article>
    `;
  }).join('');
}

function jumpToMetric(metric) {
  state.leaderboardMetric = metric;
  document.getElementById('leaderboardMetric').value = metric;
  renderLeaderboard();
  document.getElementById('leaderboard').scrollIntoView({ behavior: 'smooth' });
}

function renderBalancedModels() {
  const container = document.getElementById('balancedModels');
  const rows = MODEL_DATA.map(row => {
    const vals = ['public', 'exam', 'trading'].map(k => row.metrics[k]).filter(v => v != null);
    return {
      model: row.model,
      mean: mean(vals),
      min: Math.min(...vals),
      spread: std(vals),
    };
  }).sort((a, b) => b.mean - a.mean).slice(0, 5);

  container.innerHTML = rows.map(row => `
    <div class="balanced-item">
      <div class="balanced-item-title">${row.model}</div>
      <div class="balanced-metrics">
        <div><span>mean</span><strong>${pct(row.mean)}</strong></div>
        <div><span>min</span><strong>${pct(row.min)}</strong></div>
        <div><span>std</span><strong>${(row.spread * 100).toFixed(2)}</strong></div>
      </div>
    </div>
  `).join('');
}

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboardBody');
  const metric = state.leaderboardMetric;
  const query = state.leaderboardSearch.trim().toLowerCase();

  let rows = MODEL_DATA
    .filter(row => row.metrics[metric] != null)
    .filter(row => row.model.toLowerCase().includes(query))
    .sort((a, b) => b.metrics[metric] - a.metrics[metric]);

  tbody.innerHTML = rows.map((row, idx) => {
    const rankClass =
      idx === 0 ? 'rank-gold' :
      idx === 1 ? 'rank-silver' :
      idx === 2 ? 'rank-bronze' : '';

    const ci = getCI(row.model, metric);
    const ciText = ci ? `${pct(ci.low)} – ${pct(ci.high)}` : '—';

    const dPublicExam = delta(row.metrics.public, row.metrics.exam);
    const dPublicTrading = delta(row.metrics.public, row.metrics.trading);

    return `
      <tr>
        <td class="${rankClass}">${idx + 1}</td>
        <td>${row.model}</td>
        <td class="font-semibold">${pct(row.metrics[metric])}</td>
        <td>${ciText}</td>
        <td>${dPublicExam == null ? '—' : (dPublicExam * 100).toFixed(2)}</td>
        <td>${dPublicTrading == null ? '—' : (dPublicTrading * 100).toFixed(2)}</td>
      </tr>
    `;
  }).join('');
}

function renderModelPicker() {
  const container = document.getElementById('modelPicker');
  const rows = [...MODEL_DATA].sort((a, b) => b.metrics.public - a.metrics.public);

  container.innerHTML = rows.map(row => {
    const checked = state.selectedModels.includes(row.model);
    return `
      <label class="model-option">
        <input type="checkbox" value="${row.model}" ${checked ? 'checked' : ''} />
        <div class="flex-1 min-w-0">
          <div class="text-sm text-gray-900 truncate">${row.model}</div>
          <div class="meta">public ${pct(row.metrics.public)} · exam ${pct(row.metrics.exam)} · TA ${pct(row.metrics.trading)}</div>
        </div>
      </label>
    `;
  }).join('');

  container.querySelectorAll('input[type="checkbox"]').forEach(input => {
    input.addEventListener('change', e => {
      const model = e.target.value;
      if (e.target.checked) {
        if (state.selectedModels.length >= 10) {
          e.target.checked = false;
          alert('You can compare up to 10 models.');
          return;
        }
        state.selectedModels.push(model);
      } else {
        state.selectedModels = state.selectedModels.filter(x => x !== model);
      }
      updateSelectedCount();
      renderCompareChart();
      renderTransferGapChart();
      renderDifficultyChart();
    });
  });

  updateSelectedCount();
}

function updateSelectedCount() {
  document.getElementById('selectedCount').textContent = `${state.selectedModels.length} / 10`;
}

function renderCompareChart() {
  const metric = state.compareMetric;
  const rows = state.selectedModels
    .map(model => ({
      model,
      score: getMetricValue(model, metric),
      ci: getCI(model, metric),
    }))
    .filter(row => row.score != null)
    .sort((a, b) => b.score - a.score);

  const showCI = state.showCI;
  const errorBars = rows.map(row => row.ci ? { low: row.ci.low, high: row.ci.high } : null);
  const hasAnyCI = errorBars.some(Boolean);

  const ciNote = document.getElementById('ciNote');
  ciNote.classList.toggle('hidden', !(showCI && !hasAnyCI));

  if (compareChartInstance) compareChartInstance.destroy();

  compareChartInstance = new Chart(document.getElementById('compareChart'), {
    type: 'bar',
    data: {
      labels: rows.map(r => r.model),
      datasets: [{
        label: metricLabel(metric),
        data: rows.map(r => r.score),
        backgroundColor: rows.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'CC'),
        borderColor: rows.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        errorBars: {
          enabled: showCI && hasAnyCI,
          errorBars,
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const row = rows[ctx.dataIndex];
              if (showCI && row.ci) {
                return `${pct(row.score)} (CI ${pct(row.ci.low)} – ${pct(row.ci.high)})`;
              }
              return pct(row.score);
            },
          },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 1,
          grid: { color: '#f3f4f6' },
          ticks: {
            callback: value => `${Math.round(value * 100)}%`,
          },
        },
        x: {
          grid: { display: false },
          ticks: { maxRotation: 40, minRotation: 20 },
        },
      },
    },
  });
}

function renderTransferGapChart() {
  const selectedSet = new Set(state.selectedModels);

  const points = MODEL_DATA.map(row => ({
    x: +(row.metrics.public - row.metrics.exam).toFixed(4),
    y: +(row.metrics.public - row.metrics.trading).toFixed(4),
    label: row.model,
    selected: selectedSet.has(row.model),
  }));

  if (transferGapChartInstance) transferGapChartInstance.destroy();

  transferGapChartInstance = new Chart(document.getElementById('transferGapChart'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'All models',
          data: points.filter(p => !p.selected),
          pointRadius: 4,
          pointHoverRadius: 6,
          backgroundColor: '#9ca3af99',
          borderColor: '#9ca3af',
        },
        {
          label: 'Selected models',
          data: points.filter(p => p.selected),
          pointRadius: 6,
          pointHoverRadius: 8,
          backgroundColor: '#2563ebcc',
          borderColor: '#2563eb',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => {
              const raw = ctx.raw;
              return `${raw.label}: Δexam ${(raw.x * 100).toFixed(2)}, ΔTA ${(raw.y * 100).toFixed(2)}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Δ public → exam' },
          grid: { color: '#f3f4f6' },
          ticks: { callback: value => `${(value * 100).toFixed(0)}` },
        },
        y: {
          title: { display: true, text: 'Δ public → trading / TA' },
          grid: { color: '#f3f4f6' },
          ticks: { callback: value => `${(value * 100).toFixed(0)}` },
        },
      },
    },
  });
}

function renderDifficultyChart() {
  const selected = state.selectedModels.slice(0, 6);
  const labels = ['CFA-like Level 1', 'CFA-like Level 2', 'CFA-like Level 3'];

  if (difficultyChartInstance) difficultyChartInstance.destroy();

  difficultyChartInstance = new Chart(document.getElementById('difficultyChart'), {
    type: 'line',
    data: {
      labels,
      datasets: selected.map((model, i) => ({
        label: model,
        data: labels.map(label => getMetricValue(model, label)),
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '20',
        borderWidth: 2.5,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.25,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${pct(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 1,
          grid: { color: '#f3f4f6' },
          ticks: { callback: value => `${Math.round(value * 100)}%` },
        },
        x: {
          grid: { display: false },
        },
      },
    },
  });
}

function renderFindings() {
  const container = document.getElementById('findingCards');
  container.innerHTML = FINDINGS.map((item, i) => `
    <article class="finding-card">
      <div class="finding-number">${String(i + 1).padStart(2, '0')}</div>
      <h3 class="font-semibold text-gray-900">${item.title}</h3>
      <p class="mt-2 text-sm text-gray-600 leading-6">${item.body}</p>
    </article>
  `).join('');
}

function renderDatasetExplorer() {
  const container = document.getElementById('datasetExplorer');
  container.innerHTML = DATASET_META.map((d, i) => `
    <div class="accordion ${i === 0 ? 'open' : ''}">
      <button class="accordion-header" onclick="toggleAccordion(this)">
        <span>${d.name}</span>
        <svg class="accordion-chevron w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M19 9l-7 7-7-7"/>
        </svg>
      </button>
      <div class="accordion-body">
        <div class="dataset-badges">
          <span class="dataset-badge">${d.count} questions</span>
          <span class="dataset-badge gray">${d.format}</span>
          <span class="dataset-badge gray">${d.language}</span>
        </div>
        <div class="dataset-body">${d.description}</div>
      </div>
    </div>
  `).join('');
}

function toggleAccordion(btn) {
  const acc = btn.closest('.accordion');
  acc.classList.toggle('open');
}

function bindControls() {
  populateSelect('leaderboardMetric', state.leaderboardMetric);
  populateSelect('compareMetric', state.compareMetric);

  document.getElementById('leaderboardMetric').addEventListener('change', e => {
    state.leaderboardMetric = e.target.value;
    renderLeaderboard();
  });

  document.getElementById('leaderboardSearch').addEventListener('input', e => {
    state.leaderboardSearch = e.target.value;
    renderLeaderboard();
  });

  document.getElementById('leaderboardReset').addEventListener('click', () => {
    state.leaderboardMetric = 'public';
    state.leaderboardSearch = '';
    document.getElementById('leaderboardMetric').value = 'public';
    document.getElementById('leaderboardSearch').value = '';
    renderLeaderboard();
  });

  document.getElementById('compareMetric').addEventListener('change', e => {
    state.compareMetric = e.target.value;
    renderCompareChart();
  });

  document.getElementById('compareCI').addEventListener('change', e => {
    state.showCI = e.target.checked;
    renderCompareChart();
  });

  document.getElementById('compareTop5').addEventListener('click', () => {
    state.selectedModels = getTop5Overall();
    renderModelPicker();
    renderCompareChart();
    renderTransferGapChart();
    renderDifficultyChart();
  });

  document.getElementById('compareReset').addEventListener('click', () => {
    state.selectedModels = [];
    renderModelPicker();
    renderCompareChart();
    renderTransferGapChart();
    renderDifficultyChart();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderGroupCards();
  renderBalancedModels();
  renderFindings();
  renderDatasetExplorer();
  bindControls();
  renderModelPicker();
  renderLeaderboard();
  renderCompareChart();
  renderTransferGapChart();
  renderDifficultyChart();
});
