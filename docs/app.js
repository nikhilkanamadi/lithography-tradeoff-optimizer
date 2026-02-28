/* ================================================================
   LTO Dashboard — GitHub Pages Version (Mock Data Fallback)
   When API unavailable, uses realistic demo data
   ================================================================ */

// Try live API first, fall back to demo mode
let API = '';
let DEMO_MODE = true;

// ---- Mock Data ----
const MOCK_PREDICTIONS = {
    predictions: {
        speed_vs_accuracy: { score: 0.900, ci_low: 0.818, ci_high: 0.983 },
        resolution_vs_dof: { score: 0.744, ci_low: 0.729, ci_high: 0.759 },
        cost_vs_fidelity: { score: 0.402, ci_low: 0.271, ci_high: 0.533 },
        surrogate_reliability: { score: 0.989, ci_low: 0.980, ci_high: 0.998 },
        yield_risk: { score: 0.201, ci_low: 0.164, ci_high: 0.239 },
    },
    uncertainty: { high_uncertainty: false, confidence_level: 'medium', recommend_physics_simulation: false },
    constraints: { all_satisfied: true, violations: [] },
    inference_time_ms: 287,
    model_version: 'v0.2-ensemble',
};

const MOCK_JOB = {
    job_id: 'lto-demo-' + Date.now().toString(36),
    status: 'completed',
    outputs: {
        resolution_nm: 20.45, depth_of_focus_nm: 125.4,
        pattern_fidelity: 0.876, accuracy_vs_physics: 0.934,
        yield_prediction: 0.812, compute_time_s: 0.342,
    },
    tradeoff_signals: {
        speed_vs_accuracy: 0.900, resolution_vs_dof: 0.744,
        cost_vs_fidelity: 0.402, surrogate_reliability: 0.989,
        yield_risk: 0.201, overall_health: 0.650,
    },
    ml_prediction: { uncertainty: 'medium', constraints_satisfied: true },
};

// ---- Generate parameter-responsive mock data ----
function generateMockPrediction(na, wavelength, dose, sigma) {
    const base = { ...MOCK_PREDICTIONS };
    const p = JSON.parse(JSON.stringify(base.predictions));

    // Make scores responsive to parameter changes
    const naFactor = na / 0.33;
    const doseFactor = dose / 15.0;
    const sigmaFactor = sigma / 0.8;
    const wlFactor = wavelength < 50 ? 1.0 : wavelength < 200 ? 0.85 : 0.7;

    p.speed_vs_accuracy.score = clamp(0.9 * naFactor * wlFactor + jitter(), 0.1, 1.0);
    p.resolution_vs_dof.score = clamp(0.744 / naFactor * wlFactor + jitter(), 0.1, 1.0);
    p.cost_vs_fidelity.score = clamp(0.402 * doseFactor * sigmaFactor + jitter(), 0.1, 1.0);
    p.surrogate_reliability.score = clamp(0.989 - Math.abs(na - 0.33) * 0.2 + jitter() * 0.1, 0.7, 1.0);
    p.yield_risk.score = clamp(0.201 / naFactor / doseFactor + jitter(), 0.01, 0.99);

    Object.values(p).forEach(s => {
        s.ci_low = clamp(s.score - 0.05 - Math.random() * 0.08, 0, 1);
        s.ci_high = clamp(s.score + 0.05 + Math.random() * 0.08, 0, 1);
    });

    return {
        ...base,
        predictions: p,
        inference_time_ms: 200 + Math.random() * 200,
        uncertainty: {
            ...base.uncertainty,
            confidence_level: p.surrogate_reliability.score > 0.95 ? 'high' : p.surrogate_reliability.score > 0.85 ? 'medium' : 'low',
            high_uncertainty: p.surrogate_reliability.score < 0.85,
        },
    };
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function jitter() { return (Math.random() - 0.5) * 0.08; }

// ---- Scroll-Based Navigation ----
function scrollToSection(id) {
    document.getElementById(`section-${id}`).scrollIntoView({ behavior: 'smooth' });
}

document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => scrollToSection(btn.dataset.section));
});

const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-link');
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.id.replace('section-', '');
            navLinks.forEach(l => l.classList.remove('active'));
            const active = document.querySelector(`.nav-link[data-section="${id}"]`);
            if (active) active.classList.add('active');
        }
    });
}, { threshold: 0.4 });
sections.forEach(s => observer.observe(s));

// ---- Range Sliders ----
function bindSlider(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const disp = document.getElementById(`${id}-val`);
    if (!disp) return;
    el.addEventListener('input', () => {
        const v = parseFloat(el.value);
        disp.textContent = id.includes('resist') ? v.toFixed(0) : id.includes('dose') ? v.toFixed(1) : v.toFixed(2);
    });
}
['p-na', 'p-dose', 'p-sigma', 'p-resist', 'j-na', 'j-dose'].forEach(bindSlider);

// ---- Chart Instances ----
let radarChart = null, barChart = null, predictRadar = null;

function initCharts() {
    const rCtx = document.getElementById('radar-chart');
    if (!rCtx) return;
    radarChart = new Chart(rCtx, {
        type: 'radar',
        data: {
            labels: ['Speed / Accuracy', 'Resolution / DoF', 'Cost / Fidelity', 'Surrogate', 'Yield Safety'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                borderColor: '#667eea', backgroundColor: 'rgba(102,126,234,0.12)',
                borderWidth: 2, pointBackgroundColor: '#667eea',
                pointBorderColor: '#f0f2f8', pointBorderWidth: 1, pointRadius: 4,
            }]
        },
        options: radarOpts()
    });

    const bCtx = document.getElementById('bar-chart');
    if (!bCtx) return;
    barChart = new Chart(bCtx, {
        type: 'bar',
        data: {
            labels: ['Speed vs Accuracy', 'Resolution vs DoF', 'Cost vs Fidelity', 'Surrogate', 'Yield Risk'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: ['rgba(102,126,234,0.6)', 'rgba(6,182,212,0.6)', 'rgba(139,92,246,0.6)', 'rgba(16,185,129,0.6)', 'rgba(239,68,68,0.6)'],
                borderColor: ['#667eea', '#06b6d4', '#8b5cf6', '#10b981', '#ef4444'],
                borderWidth: 1, borderRadius: 8,
            }]
        },
        options: barOpts()
    });
}

function radarOpts() {
    return {
        responsive: true, maintainAspectRatio: false,
        scales: { r: { beginAtZero: true, max: 1, ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.05)' }, angleLines: { color: 'rgba(255,255,255,0.05)' }, pointLabels: { color: '#a0a8c0', font: { size: 11, family: 'Inter' } } } },
        plugins: { legend: { display: false } }
    };
}
function barOpts() {
    return {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5c6484', font: { size: 10 } } }, x: { grid: { display: false }, ticks: { color: '#a0a8c0', font: { size: 9 } } } },
        plugins: { legend: { display: false } }
    };
}

// ---- Fill overview from data ----
function fillOverview(d) {
    const p = d.predictions || {};
    const names = Object.keys(p);
    const scores = names.map(n => p[n].score);

    if (radarChart) {
        radarChart.data.datasets[0].data = [
            p.speed_vs_accuracy?.score || 0, p.resolution_vs_dof?.score || 0,
            p.cost_vs_fidelity?.score || 0, p.surrogate_reliability?.score || 0,
            1 - (p.yield_risk?.score || 0),
        ];
        radarChart.update('none');
    }
    if (barChart) { barChart.data.datasets[0].data = scores; barChart.update('none'); }

    const sigColors = { speed_vs_accuracy: '#667eea', resolution_vs_dof: '#06b6d4', cost_vs_fidelity: '#8b5cf6', surrogate_reliability: '#10b981', yield_risk: '#ef4444' };
    document.getElementById('signal-cards').innerHTML = names.map(n => {
        const s = p[n]; const c = sigColors[n] || '#667eea';
        const label = n.replace(/_/g, ' ').replace(/\b\w/g, x => x.toUpperCase());
        return `<div class="signal-card glass-card"><div class="sig-name">${label}</div><div class="sig-score" style="color:${c}">${s.score.toFixed(3)}</div><div class="sig-ci">[${s.ci_low.toFixed(3)}, ${s.ci_high.toFixed(3)}]</div></div>`;
    }).join('');
}

// ---- Health ----
function renderHealth() {
    const items = [
        { icon: '🚀', name: 'FastAPI Server', status: DEMO_MODE ? 'demo' : 'healthy' },
        { icon: '🧠', name: 'ML Ensemble', status: DEMO_MODE ? 'demo' : 'loaded' },
        { icon: '🔬', name: 'Simulator', status: DEMO_MODE ? 'demo' : 'available' },
    ];
    document.getElementById('health-grid').innerHTML = items.map(i => {
        const up = !DEMO_MODE;
        const cls = DEMO_MODE ? 'h-demo' : 'h-up';
        return `<div class="health-card glass-card"><div class="health-icon">${i.icon}</div><div class="health-name">${i.name}</div><div class="health-status ${cls}">${DEMO_MODE ? '⬡ ' : '✓ '}${i.status}</div></div>`;
    }).join('');
}

// ---- Predict ----
async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.classList.add('loading'); btn.textContent = '⏳ Predicting...';

    const na = +document.getElementById('p-na').value;
    const wavelength = +document.getElementById('p-wavelength').value;
    const dose = +document.getElementById('p-dose').value;
    const sigma = +document.getElementById('p-sigma').value;

    let d;
    if (!DEMO_MODE) {
        try {
            const r = await fetch(`${API}/api/v1/tradeoffs/predict`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ na, wavelength_nm: wavelength, dose_mj_cm2: dose, sigma, resist_thickness_nm: +document.getElementById('p-resist').value, grid_size_nm: +document.getElementById('p-grid').value, use_ai_surrogate: document.getElementById('p-surrogate').checked })
            });
            d = await r.json();
        } catch { d = generateMockPrediction(na, wavelength, dose, sigma); }
    } else {
        await new Promise(r => setTimeout(r, 300 + Math.random() * 400));
        d = generateMockPrediction(na, wavelength, dose, sigma);
    }
    showPrediction(d);

    btn.classList.remove('loading');
    btn.innerHTML = 'Predict Tradeoffs <span class="btn-arrow">→</span>';
}

function showPrediction(d) {
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-live').classList.remove('hidden');

    const p = d.predictions || {}, u = d.uncertainty || {}, c = d.constraints || {};
    const names = Object.keys(p);
    const colors = ['#667eea', '#06b6d4', '#8b5cf6', '#10b981', '#ef4444'];

    const confCls = `conf-${u.confidence_level || 'medium'}`;
    document.getElementById('result-badges').innerHTML =
        `<span class="rbadge ${confCls}">Confidence: ${u.confidence_level || '—'}</span>` +
        `<span class="rbadge time">⏱ ${(d.inference_time_ms || 0).toFixed(0)}ms</span>` +
        `<span class="rbadge version">📦 ${d.model_version || 'v0.2'}</span>` +
        (DEMO_MODE ? '<span class="rbadge demo">🎮 Demo Mode</span>' : '');

    document.getElementById('result-grid').innerHTML = names.map((n, i) => {
        const s = p[n]; const label = n.replace(/_/g, ' ');
        return `<div class="score-tile"><div class="tile-name">${label}</div><div class="tile-score" style="color:${colors[i % 5]}">${s.score.toFixed(3)}</div><div class="tile-ci">[${s.ci_low.toFixed(3)}, ${s.ci_high.toFixed(3)}]</div></div>`;
    }).join('');

    const ctx = document.getElementById('predict-radar');
    if (predictRadar) predictRadar.destroy();
    predictRadar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: names.map(n => n.replace(/_/g, ' ')),
            datasets: [
                { data: names.map(n => p[n].score), borderColor: '#667eea', backgroundColor: 'rgba(102,126,234,0.12)', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#667eea' },
                { data: names.map(n => p[n].ci_low), borderColor: 'rgba(102,126,234,0.25)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
                { data: names.map(n => p[n].ci_high), borderColor: 'rgba(102,126,234,0.25)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
            ]
        },
        options: radarOpts()
    });

    const uncClass = u.high_uncertainty ? 'info-bad' : u.confidence_level === 'high' ? 'info-ok' : 'info-warn';
    const conClass = c.all_satisfied ? 'info-ok' : 'info-bad';
    document.getElementById('result-info').innerHTML =
        `<div class="info-box ${uncClass}"><strong>${u.high_uncertainty ? '⚠️ High Uncertainty' : '✓ Uncertainty Normal'}</strong>${u.recommend_physics_simulation ? '<br>💡 Recommend full physics simulation' : ''}</div>` +
        `<div class="info-box ${conClass}">${c.all_satisfied ? '<strong>✓ All Constraints Satisfied</strong>' : `<strong>⚠️ Violations</strong><br>${(c.violations || []).join('<br>')}`}</div>`;
}

// ---- Submit Job ----
async function submitJob() {
    const btn = document.getElementById('btn-submit');
    btn.classList.add('loading'); btn.textContent = '⏳ Running pipeline...';

    let d;
    if (!DEMO_MODE) {
        try {
            const r = await fetch(`${API}/api/v1/jobs/submit`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ na: +document.getElementById('j-na').value, wavelength_nm: +document.getElementById('j-wavelength').value, dose_mj_cm2: +document.getElementById('j-dose').value, sigma: 0.8, grid_size_nm: 1.0, use_ai_surrogate: document.getElementById('j-surrogate').checked, pattern_complexity: document.getElementById('j-complexity').value })
            });
            d = await r.json();
        } catch { d = { ...MOCK_JOB, job_id: 'lto-demo-' + Date.now().toString(36) }; }
    } else {
        await new Promise(r => setTimeout(r, 500 + Math.random() * 600));
        d = { ...MOCK_JOB, job_id: 'lto-demo-' + Date.now().toString(36) };
    }
    showJob(d);

    btn.classList.remove('loading');
    btn.innerHTML = 'Submit Job <span class="btn-arrow">→</span>';
}

function showJob(d) {
    document.getElementById('job-empty').classList.add('hidden');
    document.getElementById('job-live').classList.remove('hidden');

    document.getElementById('job-header-info').innerHTML =
        `<div style="font-size:1rem;font-weight:600;color:var(--accent);margin-bottom:4px">Job: ${d.job_id}</div>` +
        `<div style="font-size:0.78rem;color:var(--green);font-weight:600">● ${d.status}${DEMO_MODE ? ' (demo)' : ''}</div>`;

    const o = d.outputs || {};
    document.getElementById('job-output-grid').innerHTML =
        `<div class="job-section-title">Simulation Outputs</div>` +
        `<div class="job-output-grid">
            <div class="job-tile"><div class="job-tile-label">Resolution</div><div class="job-tile-val">${(o.resolution_nm || 0).toFixed(2)}</div><div class="job-tile-unit">nm</div></div>
            <div class="job-tile"><div class="job-tile-label">Depth of Focus</div><div class="job-tile-val">${(o.depth_of_focus_nm || 0).toFixed(1)}</div><div class="job-tile-unit">nm</div></div>
            <div class="job-tile"><div class="job-tile-label">Pattern Fidelity</div><div class="job-tile-val">${(o.pattern_fidelity || 0).toFixed(3)}</div></div>
            <div class="job-tile"><div class="job-tile-label">Accuracy vs Physics</div><div class="job-tile-val">${(o.accuracy_vs_physics || 0).toFixed(3)}</div></div>
            <div class="job-tile"><div class="job-tile-label">Yield Prediction</div><div class="job-tile-val">${(o.yield_prediction || 0).toFixed(3)}</div></div>
            <div class="job-tile"><div class="job-tile-label">Compute Time</div><div class="job-tile-val">${(o.compute_time_s || 0).toFixed(3)}</div><div class="job-tile-unit">s</div></div>
        </div>`;

    const sig = d.tradeoff_signals || {};
    const sigColors = { speed_vs_accuracy: '#667eea', resolution_vs_dof: '#06b6d4', cost_vs_fidelity: '#8b5cf6', surrogate_reliability: '#10b981', yield_risk: '#ef4444', overall_health: '#e8d44d' };
    document.getElementById('job-signal-grid').innerHTML =
        `<div class="job-section-title">Tradeoff Signals</div>` +
        `<div class="job-output-grid">${Object.keys(sig).map(k => {
            const c = sigColors[k] || '#667eea'; const label = k.replace(/_/g, ' ');
            return `<div class="job-tile"><div class="job-tile-label">${label}</div><div class="job-tile-val" style="color:${c}">${sig[k].toFixed(3)}</div></div>`;
        }).join('')}</div>`;

    const ml = d.ml_prediction || {};
    document.getElementById('job-ml-info').innerHTML =
        `<div class="job-section-title">ML Prediction</div>` +
        `<div style="display:flex;gap:8px;flex-wrap:wrap">` +
        `<span class="rbadge conf-${ml.uncertainty || 'medium'}">Confidence: ${ml.uncertainty || '—'}</span>` +
        `<span class="rbadge ${ml.constraints_satisfied ? 'conf-high' : 'conf-low'}">Constraints: ${ml.constraints_satisfied ? '✓ Satisfied' : '✗ Violated'}</span>` +
        `</div>`;
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', async () => {
    initCharts();

    // Try to detect live API
    try {
        const r = await fetch('http://127.0.0.1:8000/api/v1/health', { signal: AbortSignal.timeout(2000) });
        if (r.ok) { API = 'http://127.0.0.1:8000'; DEMO_MODE = false; }
    } catch { /* stay in demo mode */ }

    if (!DEMO_MODE) {
        document.querySelector('.nav-status span:last-child').textContent = 'Connected';
        document.querySelector('.status-dot').className = 'status-dot live';
    }

    // Load overview data
    fillOverview(MOCK_PREDICTIONS);
    renderHealth();
});
