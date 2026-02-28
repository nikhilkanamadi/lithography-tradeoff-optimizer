/* ================================================================
   LTO Dashboard — Application Logic (on.energy inspired)
   Scroll-based navigation, API integration, Chart.js
   ================================================================ */

const API = 'http://127.0.0.1:8000';

// ---- Scroll-Based Navigation ----
function scrollToSection(id) {
    document.getElementById(`section-${id}`).scrollIntoView({ behavior: 'smooth' });
}

document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => scrollToSection(btn.dataset.section));
});

// Highlight active nav on scroll
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
                borderColor: '#667eea',
                backgroundColor: 'rgba(102,126,234,0.12)',
                borderWidth: 2,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#f0f2f8',
                pointBorderWidth: 1,
                pointRadius: 4,
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

// ---- API: Health ----
async function checkHealth() {
    try {
        const r = await fetch(`${API}/api/v1/health`);
        const d = await r.json();
        const dot = document.querySelector('.status-dot');
        const txt = document.querySelector('.nav-status span:last-child');
        if (d.status === 'healthy') {
            dot.className = 'status-dot live'; txt.textContent = 'Connected';
        } else {
            dot.className = 'status-dot dead'; txt.textContent = 'Degraded';
        }
        renderHealth(d);
    } catch {
        document.querySelector('.status-dot').className = 'status-dot dead';
        document.querySelector('.nav-status span:last-child').textContent = 'Offline';
    }
}

function renderHealth(d) {
    const c = d.components || {};
    const items = [
        { icon: '🚀', name: 'FastAPI Server', status: c.api || 'unknown' },
        { icon: '🧠', name: 'ML Ensemble', status: c.model || 'unknown' },
        { icon: '🔬', name: 'Simulator', status: c.simulator || 'unknown' },
    ];
    document.getElementById('health-grid').innerHTML = items.map(i => {
        const up = ['healthy', 'loaded', 'available'].includes(i.status);
        return `<div class="health-card glass-card"><div class="health-icon">${i.icon}</div><div class="health-name">${i.name}</div><div class="health-status ${up ? 'h-up' : 'h-down'}">${up ? '✓ ' : '✗ '}${i.status}</div></div>`;
    }).join('');
}

// ---- API: Overview data ----
async function loadOverview() {
    try {
        const r = await fetch(`${API}/api/v1/tradeoffs/predict`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ na: 0.33, wavelength_nm: 13.5, dose_mj_cm2: 15.0, sigma: 0.8, grid_size_nm: 1.0, use_ai_surrogate: true })
        });
        const d = await r.json();
        fillOverview(d);
    } catch (e) { console.error('Overview load failed:', e); }
}

function fillOverview(d) {
    const p = d.predictions || {};
    const names = Object.keys(p);
    const scores = names.map(n => p[n].score);

    // Hero stats
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const setVal = (id, val) => { const el = document.querySelector(`#${id} .stat-value`); if (el) el.textContent = val; };
    setVal('stat-health', (avg * 100).toFixed(0));
    setVal('stat-risk', (p.yield_risk?.score || 0).toFixed(3));
    setVal('stat-reliability', (p.surrogate_reliability?.score || 0).toFixed(3));

    // Radar
    if (radarChart) {
        radarChart.data.datasets[0].data = [
            p.speed_vs_accuracy?.score || 0, p.resolution_vs_dof?.score || 0,
            p.cost_vs_fidelity?.score || 0, p.surrogate_reliability?.score || 0,
            1 - (p.yield_risk?.score || 0),
        ];
        radarChart.update('none');
    }
    if (barChart) { barChart.data.datasets[0].data = scores; barChart.update('none'); }

    // Signal cards
    const sigColors = { speed_vs_accuracy: '#667eea', resolution_vs_dof: '#06b6d4', cost_vs_fidelity: '#8b5cf6', surrogate_reliability: '#10b981', yield_risk: '#ef4444' };
    document.getElementById('signal-cards').innerHTML = names.map(n => {
        const s = p[n]; const c = sigColors[n] || '#667eea';
        const label = n.replace(/_/g, ' ').replace(/\b\w/g, x => x.toUpperCase());
        return `<div class="signal-card glass-card"><div class="sig-name">${label}</div><div class="sig-score" style="color:${c}">${s.score.toFixed(3)}</div><div class="sig-ci">[${s.ci_low.toFixed(3)}, ${s.ci_high.toFixed(3)}]</div></div>`;
    }).join('');
}

// ---- Predict ----
async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.classList.add('loading'); btn.textContent = '⏳ Predicting...';

    const body = {
        na: +document.getElementById('p-na').value,
        wavelength_nm: +document.getElementById('p-wavelength').value,
        dose_mj_cm2: +document.getElementById('p-dose').value,
        sigma: +document.getElementById('p-sigma').value,
        resist_thickness_nm: +document.getElementById('p-resist').value,
        grid_size_nm: +document.getElementById('p-grid').value,
        use_ai_surrogate: document.getElementById('p-surrogate').checked,
    };

    try {
        const r = await fetch(`${API}/api/v1/tradeoffs/predict`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        showPrediction(await r.json());
    } catch (e) { alert('Prediction failed: ' + e.message); }

    btn.classList.remove('loading');
    btn.innerHTML = 'Predict Tradeoffs <span class="btn-arrow">→</span>';
}

function showPrediction(d) {
    document.getElementById('results-empty').classList.add('hidden');
    document.getElementById('results-live').classList.remove('hidden');

    const p = d.predictions || {}, u = d.uncertainty || {}, c = d.constraints || {};
    const names = Object.keys(p);
    const colors = ['#667eea', '#06b6d4', '#8b5cf6', '#10b981', '#ef4444'];

    // Badges
    const confCls = `conf-${u.confidence_level || 'medium'}`;
    document.getElementById('result-badges').innerHTML =
        `<span class="rbadge ${confCls}">Confidence: ${u.confidence_level || '—'}</span>` +
        `<span class="rbadge time">⏱ ${(d.inference_time_ms || 0).toFixed(0)}ms</span>` +
        `<span class="rbadge version">📦 ${d.model_version || 'v0.2'}</span>`;

    // Score tiles
    document.getElementById('result-grid').innerHTML = names.map((n, i) => {
        const s = p[n]; const label = n.replace(/_/g, ' ');
        return `<div class="score-tile"><div class="tile-name">${label}</div><div class="tile-score" style="color:${colors[i % 5]}">${s.score.toFixed(3)}</div><div class="tile-ci">[${s.ci_low.toFixed(3)}, ${s.ci_high.toFixed(3)}]</div></div>`;
    }).join('');

    // Radar
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

    // Info boxes
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

    const body = {
        na: +document.getElementById('j-na').value,
        wavelength_nm: +document.getElementById('j-wavelength').value,
        dose_mj_cm2: +document.getElementById('j-dose').value,
        sigma: 0.8, grid_size_nm: 1.0,
        use_ai_surrogate: document.getElementById('j-surrogate').checked,
        pattern_complexity: document.getElementById('j-complexity').value,
    };

    try {
        const r = await fetch(`${API}/api/v1/jobs/submit`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        showJob(await r.json());
    } catch (e) { alert('Job failed: ' + e.message); }

    btn.classList.remove('loading');
    btn.innerHTML = 'Submit Job <span class="btn-arrow">→</span>';
}

function showJob(d) {
    document.getElementById('job-empty').classList.add('hidden');
    document.getElementById('job-live').classList.remove('hidden');

    document.getElementById('job-header-info').innerHTML =
        `<div style="font-size:1rem;font-weight:600;color:var(--accent);margin-bottom:4px">Job: ${d.job_id}</div>` +
        `<div style="font-size:0.78rem;color:var(--green);font-weight:600">● ${d.status}</div>`;

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
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    checkHealth();
    loadOverview();
    setInterval(checkHealth, 30000);
});
