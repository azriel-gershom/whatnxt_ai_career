// ═══════════════════════════════════════════════════════════════
// PASTE THIS to REPLACE the matching functions in your <script>
// ═══════════════════════════════════════════════════════════════

// ── 1. YOUTUBE SKILL PILLS — one pill per CSV row's skill set ──
// Replace the existing YT_SKILL_SHORTCUTS constant and buildYtSkillPills + loadYtVideos functions

const YT_SKILL_SHORTCUTS = {
  'Software Engineer':       [['Python & DSA','Python'],['Java & Spring Boot','Java'],['React & Node.js','React']],
  'Data Scientist':          [['Pandas & Stats','Pandas'],['SQL & Tableau','SQL'],['Deep Learning & NLP','TensorFlow']],
  'Cybersecurity Analyst':   [['Networking & Linux','Linux Security'],['Ethical Hacking','Kali Linux'],['SIEM & Splunk','Splunk']],
  'Cloud Architect':         [['AWS Fundamentals','AWS'],['Azure & DevOps','Azure'],['GCP & Kubernetes','Kubernetes']],
  'AI Engineer':             [['PyTorch & LLMs','PyTorch'],['LangChain & RAG','LangChain'],['MLOps & Docker','MLOps']],
};

// Builds one pill per skill row — clicking loads the matching video set
function buildYtSkillPills(role) {
  const pills = $('yt-skill-pills');
  if (!pills) return;
  const tracks = YT_SKILL_SHORTCUTS[role] || [[role, role]];
  pills.innerHTML = tracks.map(([label, skill]) =>
    `<button class="skill-pill-search" onclick="loadYtVideos('${skill}','${role}',this)">${label}</button>`
  ).join('') +
  `<button class="skill-pill-search" onclick="loadYtVideos('','${role}',this)">📺 Overview</button>`;
}

// Loads videos — now returns skill-matched videos from CSV, not always Python
async function loadYtVideos(skill, role, btn) {
  const grid = $('yt-grid-roadmap');
  if (!grid) return;
  document.querySelectorAll('.skill-pill-search').forEach(b => b.classList.remove('loading','yt-active'));
  if (btn) { btn.classList.add('loading'); }
  grid.innerHTML = `<div class="yt-loading"><div class="spinner" style="width:32px;height:32px;border-width:2px;"></div><br>Finding best tutorials for <b>${skill || role}</b>…</div>`;
  try {
    const res  = await fetch(`${API_BASE}/get_youtube_videos`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ skill, role })
    });
    const data = await res.json();
    const vids = data.videos || [];
    if (!vids.length) {
      grid.innerHTML = `<div class="yt-empty">No videos found for "${skill || role}". Try another track or add your YouTube API key to api.py.</div>`;
    } else {
      grid.innerHTML = vids.map(v => `
        <div class="yt-card">
          <div class="yt-iframe-wrap">
            <iframe src="${v.embed_url}?rel=0&modestbranding=1"
                    title="${(v.title||'').replace(/"/g,'&quot;')}"
                    allow="accelerometer; autoplay; encrypted-media; gyroscope"
                    allowfullscreen loading="lazy"></iframe>
          </div>
          <div class="yt-card-body">
            <div class="yt-card-title">${v.title || 'Tutorial'}</div>
            <div class="yt-card-channel">📺 ${v.channel || 'WhatNxt Curated'}</div>
            ${v.skills ? `<div style="font-size:10px;color:var(--tx-muted);margin-top:4px;font-family:var(--ff-mono);">${v.skills.split(';').slice(0,3).join(' · ')}</div>` : ''}
          </div>
        </div>`).join('');
    }
  } catch(e) {
    grid.innerHTML = `<div class="yt-empty" style="color:var(--danger);">⚠️ Could not load videos — is backend running?</div>`;
  } finally {
    if (btn) { btn.classList.remove('loading'); btn.classList.add('yt-active'); }
  }
}

// Trigger YouTube section after roadmap generates + auto-load first pill
const _origGenerateRoadmap = typeof generateRoadmap !== 'undefined' ? generateRoadmap : null;
generateRoadmap = async function() {
  if (_origGenerateRoadmap) await _origGenerateRoadmap.call(this);
  const role  = $('target-role')?.value || '';
  const ytSec = $('yt-section-roadmap');
  if (ytSec && role) {
    ytSec.style.display = 'block';
    buildYtSkillPills(role);
    // Auto-load the FIRST skill track immediately
    const firstPill = $('yt-skill-pills')?.querySelector('.skill-pill-search');
    if (firstPill) firstPill.click();
  }
};

// ── 2. JOBS — Smart Apply button (tries official career page first) ──
// Replace the entire loadJobs function

async function loadJobs() {
  const grid = $('jobs-grid'), countEl = $('jobs-count');
  if (!grid) return;
  const role = $('job-filter-role')?.value || 'All';
  const type = $('job-filter-type')?.value || 'All';
  const loc  = $('job-filter-loc')?.value  || 'All';
  grid.innerHTML = `<div class="loading-state" style="grid-column:1/-1;"><div class="spinner"></div><div class="loading-text">FETCHING OPPORTUNITIES…</div></div>`;
  try {
    const res  = await fetch(`${API_BASE}/get_jobs`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ role, type, location: loc })
    });
    const data = await res.json();
    const jobs = data.jobs || [];
    if (countEl) countEl.textContent = `${jobs.length} job${jobs.length !== 1 ? 's' : ''} found`;
    if (!jobs.length) {
      grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:40px;color:var(--tx-muted);">No jobs found. Try different filters.</div>`;
      return;
    }
    window._lastJobs = jobs;
    grid.innerHTML = jobs.map((j, i) => {
      // Smart apply URL: prefer official careers page, fallback to apply link, then LinkedIn
      const applyUrl   = j.Apply_Link || j.Company_Careers_URL || 'https://linkedin.com/jobs';
      const careersUrl = j.Company_Careers_URL || j.Apply_Link || 'https://linkedin.com/jobs';

      // Detect if it's a direct application form vs a listing page
      const isDirect = applyUrl && (
        applyUrl.includes('apply') || applyUrl.includes('application') ||
        applyUrl.includes('greenhouse') || applyUrl.includes('lever.co') ||
        applyUrl.includes('workday') || applyUrl.includes('taleo') ||
        applyUrl.includes('icims') || applyUrl.includes('smartrecruiters') ||
        applyUrl.includes('ashbyhq') || applyUrl.includes('careers.')
      );

      return `
      <div class="job-card">
        <span class="job-logo">${j.Logo_Emoji || '🏢'}</span>
        <div class="job-role">${j.Role}</div>
        <div class="job-company">${j.Company}</div>
        ${j.Industry ? `<div class="job-industry">🏭 ${j.Industry}</div>` : ''}
        <div class="job-meta">
          <span class="job-chip">${j.Location}</span>
          <span class="job-chip ${j.Type === 'Internship' ? 'green' : 'blue'}">${j.Type}</span>
          <span class="job-chip">📅 ${j.Posted || 'Recently'}</span>
        </div>
        <div class="job-salary">💰 ${j.Salary}</div>
        <div class="job-skills">🛠 ${j.Skills_Required}</div>
        <div style="display:flex;gap:7px;margin-top:10px;flex-wrap:wrap;">
          <button class="btn-apply" style="flex:2;min-width:130px;"
                  onclick="openApplyModal(window._lastJobs[${i}])">
            ✉️ AI Cover Letter
          </button>
          <a class="btn-apply" href="${applyUrl}" target="_blank" rel="noopener"
             style="flex:1;min-width:90px;background:${isDirect
               ? 'linear-gradient(135deg,#16a34a,#15803d)'
               : 'transparent'};
             ${isDirect ? '' : 'border:1.5px solid var(--border-hover);color:var(--tx-accent);'}
             text-decoration:none;"
             title="${isDirect ? 'Opens official application form' : 'Opens job listing'}">
            ${isDirect ? '🚀 Apply Now' : '🔗 View Job'}
          </a>
          ${careersUrl && careersUrl !== applyUrl ? `
          <a class="btn-apply" href="${careersUrl}" target="_blank" rel="noopener"
             style="flex:1;min-width:90px;background:transparent;border:1.5px solid var(--border-hover);color:var(--tx-accent);text-decoration:none;"
             title="Official company careers page">
            🏢 Careers
          </a>` : ''}
        </div>
      </div>`;
    }).join('');
  } catch(e) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:30px;color:var(--danger);">⚠️ Could not load jobs. Is backend running?</div>`;
  }
}
