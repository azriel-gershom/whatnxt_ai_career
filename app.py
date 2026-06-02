from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import sqlite3, pandas as pd, base64, json, re, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

# ─── Base Paths ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / 'data'
DB_DIR    = BASE_DIR / 'db'

try:
    from googleapiclient.discovery import build as yt_build
    YT_LIB_OK = True
except ImportError:
    YT_LIB_OK = False
    print("⚠️  google-api-python-client not installed.")

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════
# 🔑 API KEYS — hot-reloaded from .env on every request
# Gemini: https://aistudio.google.com/apikey
# ═══════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')

GEMINI_KEYS = [k.strip() for k in os.environ.get('GEMINI_KEYS', '').split(',') if k.strip()]
if GEMINI_KEYS:
    print(f"✅ {len(GEMINI_KEYS)} Gemini key(s) loaded (PRIMARY)")
else:
    print("⚠️  No GEMINI_KEYS found — Gemini disabled.")

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()
if OPENROUTER_API_KEY:
    print(f"✅ OpenRouter API key loaded (FALLBACK — Gemma 4 26B)")
else:
    print("ℹ️  No OPENROUTER_API_KEY — OpenRouter fallback disabled.")


YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', '')
_key_idx = 0

def _reload_keys():
    """Hot-reload API keys from .env so you never need to restart the server."""
    global GEMINI_KEYS, OPENROUTER_API_KEY
    load_dotenv(BASE_DIR / '.env', override=True)
    new_or = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if new_or and new_or != OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = new_or
        print(f"🔄 Hot-reloaded OpenRouter API key from .env")
    fresh = [k.strip() for k in os.environ.get('GEMINI_KEYS', '').split(',') if k.strip()]
    if fresh and fresh != GEMINI_KEYS:
        GEMINI_KEYS = fresh
        print(f"🔄 Hot-reloaded {len(GEMINI_KEYS)} Gemini key(s) from .env")
    return GEMINI_KEYS

def get_client():
    keys = _reload_keys()
    if not keys:
        raise Exception("No GEMINI_KEYS configured! Add keys to .env file.")
    return genai.Client(api_key=keys[_key_idx % len(keys)])

# ── Gemini models (PRIMARY) ──
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]
# Keep legacy reference
MODELS = GEMINI_MODELS

# ── OpenRouter model (FALLBACK) ──
OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it"



# ═══════════════════════════════════════════════════════════════
# 🌐 OPENROUTER CALLER — Gemma 4 26B via OpenRouter (FALLBACK)
# ═══════════════════════════════════════════════════════════════
def call_openrouter(prompt, max_output_tokens=8192):
    """Call OpenRouter API — fallback provider for Gemma 4 26B."""
    _reload_keys()
    if not OPENROUTER_API_KEY:
        raise Exception("No OPENROUTER_API_KEY configured. Get key at https://openrouter.ai/keys")
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    try:
        print(f"🔄 Trying OpenRouter {OPENROUTER_MODEL}...")
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are WhatNxt AI, an expert career guidance system. You generate extremely detailed, comprehensive, beautiful raw HTML content with ALL inline CSS. You NEVER use markdown. You ALWAYS output complete, long, detailed responses. You MUST generate ALL sections requested — never truncate or skip sections. Output raw HTML only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_output_tokens,
            temperature=0.8,
        )
        text = resp.choices[0].message.content
        if text:
            print(f"✅ Success — OpenRouter {OPENROUTER_MODEL}")
            return text
        raise Exception("Empty response from OpenRouter")
    except Exception as e:
        err = str(e)
        print(f"⚠️  OpenRouter {OPENROUTER_MODEL}: {err[:150]}")
        raise



# ═══════════════════════════════════════════════════════════════
# 🤖 SMART AI CALLER — Gemini (PRIMARY) → OpenRouter (FALLBACK)
# ═══════════════════════════════════════════════════════════════
def call_gemini(prompt, max_output_tokens=8192):
    global _key_idx
    _reload_keys()

    # ── Phase 1: Try all Gemini models (PRIMARY) ──
    total_keys = len(GEMINI_KEYS)
    if total_keys > 0:
        for model in GEMINI_MODELS:
            for attempt in range(total_keys * 2):
                try:
                    client = get_client()
                    key_num = _key_idx % total_keys
                    print(f"🔄 Trying {model} with key[{key_num}] (attempt {attempt+1})...")
                    cfg = types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=0.8,
                    )
                    if "2.5" in model:
                        cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=cfg,
                    )
                    text = resp.text
                    if text:
                        print(f"✅ Success — {model}")
                        return text
                    else:
                        print(f"⚠️  {model} returned empty response, retrying...")
                        continue
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        _key_idx += 1
                        if (_key_idx % total_keys) == 0:
                            wait = min(8, 3 * (attempt // total_keys + 1))
                            print(f"⚠️  All keys exhausted on {model} — waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"⚠️  Key[{key_num}] quota hit — rotating to key[{_key_idx % total_keys}]...")
                            time.sleep(0.5)
                    elif "503" in err or "UNAVAILABLE" in err:
                        print(f"⚠️  {model} busy, retrying in 2s...")
                        time.sleep(2)
                    elif "404" in err or "NOT_FOUND" in err:
                        print(f"⚠️  {model} not available, trying next model...")
                        break
                    else:
                        print(f"⚠️  {model}: {err[:120]}")
                        _key_idx += 1
                        time.sleep(1)

    # ── Phase 2: Try OpenRouter (Gemma 4 26B — FALLBACK) ──
    if OPENROUTER_API_KEY:
        try:
            print(f"⚠️  All Gemini models exhausted — falling back to OpenRouter...")
            return call_openrouter(prompt, max_output_tokens)
        except Exception as or_err:
            print(f"⚠️  OpenRouter fallback also failed: {or_err}")

    raise Exception("All AI providers exhausted. Add a new API key to .env")

# ─── Load Datasets ───────────────────────────────────────────
def load_csv(name):
    try:
        df = pd.read_csv(name)
        print(f"✅ {name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"⚠️  {name}: {e}")
        return pd.DataFrame()

df_career  = load_csv(DATA_DIR / 'career_data.csv')
df_courses = load_csv(DATA_DIR / 'courses_data.csv')
df_quiz    = load_csv(DATA_DIR / 'quiz_data.csv')
df_psycho  = load_csv(DATA_DIR / 'psychometric_map.csv')

# ─── Database ────────────────────────────────────────────────
DB = str(DB_DIR / 'whatnxt.db')

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT UNIQUE, password TEXT, standard TEXT, gpa REAL, goals TEXT,
        career_path TEXT, quiz_scores TEXT, gender TEXT, dob TEXT, college TEXT,
        department TEXT, maths_grade TEXT, cs_grade TEXT, physics_grade TEXT,
        english_grade TEXT, skill_level TEXT, path_choice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    existing = {row[1] for row in c.execute("PRAGMA table_info(users)")}
    for col, dt in {'gender':'TEXT','dob':'TEXT','college':'TEXT','department':'TEXT',
                    'maths_grade':'TEXT','cs_grade':'TEXT','physics_grade':'TEXT',
                    'english_grade':'TEXT','skill_level':'TEXT','path_choice':'TEXT'}.items():
        if col not in existing:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {dt}")
    c.execute('''CREATE TABLE IF NOT EXISTS progress (
        username TEXT, action TEXT, detail TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()

init_db()

def log_progress(username, action, detail=""):
    try:
        conn = get_db()
        conn.execute("INSERT INTO progress (username,action,detail) VALUES (?,?,?)",
                     (username, action, detail))
        conn.commit(); conn.close()
    except: pass

# ─── RAG helpers ─────────────────────────────────────────────
def get_career_rows(role):
    if df_career.empty: return []
    rows = df_career[df_career['Target_Role'] == role]
    return [r.to_dict() for _, r in rows.iterrows()]

def get_career_context(role):
    rows = get_career_rows(role)
    if not rows: return ""
    parts = []
    for r in rows:
        parts.append(f"""
Role: {r.get('Target_Role','')}
Skills track: {r.get('Required_Skills','')}
Year 1: {r.get('Year_1_Plan','')} | Year 2: {r.get('Year_2_Plan','')}
Year 3: {r.get('Year_3_Plan','')} | Year 4: {r.get('Year_4_Plan','')}
Companies: {r.get('Top_Companies','')} | Salary: {r.get('Salary_Range','')}
Certifications: {r.get('Certifications','')}
Demo Job: {r.get('Demo_Job_Title','')} at {r.get('Demo_Company','')}
Paid Course: {r.get('Phase_2_Course_Link','')}
Free Course 1: {r.get('Free_Course_1','')}
Free Course 2: {r.get('Free_Course_2','')}
Free Course 3: {r.get('Free_Course_3','')}""")
    return "\n---\n".join(parts)

# ─── YouTube ─────────────────────────────────────────────────
def get_videos_for_skill(role, skill=""):
    rows = get_career_rows(role)
    if not rows: return []
    best_row = rows[0]
    if skill:
        skill_lower = skill.lower()
        best_score = 0
        for r in rows:
            row_skills = r.get('Required_Skills', '').lower()
            score = sum(1 for w in skill_lower.split() if w in row_skills)
            if score > best_score:
                best_score = score
                best_row = r
    videos = []
    for embed_key in ['Phase_1_YouTube_Embed', 'Extra_YT_1', 'Extra_YT_2']:
        embed_url = best_row.get(embed_key, '')
        if not embed_url or embed_url == 'nan': continue
        if 'watch?v=' in embed_url:
            vid_id = embed_url.split('watch?v=')[-1].split('&')[0]
            embed_url = f"https://www.youtube.com/embed/{vid_id}"
        vid_id = embed_url.split('embed/')[-1].split('?')[0]
        skill_label = best_row.get('Required_Skills', role).split(';')[0].strip()
        videos.append({
            "video_id":  vid_id,
            "title":     f"{skill or skill_label} — Tutorial",
            "channel":   "WhatNxt Curated",
            "thumbnail": f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
            "embed_url": embed_url,
            "watch_url": f"https://www.youtube.com/watch?v={vid_id}",
            "skills":    best_row.get('Required_Skills','')
        })
    return videos

def search_youtube_live(query, max_results=3):
    if not YT_LIB_OK or not YOUTUBE_API_KEY: return []
    try:
        yt  = yt_build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        req = yt.search().list(part="snippet", q=query, type="video",
                               maxResults=max_results, order="relevance",
                               videoDuration="medium", relevanceLanguage="en",
                               safeSearch="strict")
        resp = req.execute()
        videos = []
        for item in resp.get('items', []):
            vid_id = item['id']['videoId']
            snip   = item['snippet']
            videos.append({
                "video_id":  vid_id,
                "title":     snip['title'],
                "channel":   snip['channelTitle'],
                "thumbnail": snip['thumbnails']['medium']['url'],
                "embed_url": f"https://www.youtube.com/embed/{vid_id}",
                "watch_url": f"https://www.youtube.com/watch?v={vid_id}",
                "skills":    query
            })
        return videos
    except Exception as e:
        print(f"YouTube API error: {e}")
        return []

# ══════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════
@app.route('/api/auth', methods=['POST'])
def auth():
    d = request.json
    username = d.get('username','').strip()
    password = d.get('password','')
    conn = get_db(); c = conn.cursor()

    if d.get('action') == 'signup':
        try:
            c.execute('''INSERT INTO users
                (username,password,standard,gpa,goals,gender,dob,college,department,
                 maths_grade,cs_grade,physics_grade,english_grade,skill_level,path_choice)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (username, password, d.get('standard',''), float(d.get('gpa') or 0),
                 d.get('goals',''), d.get('gender',''), d.get('dob',''),
                 d.get('college',''), d.get('department',''),
                 d.get('maths_grade',''), d.get('cs_grade',''),
                 d.get('physics_grade',''), d.get('english_grade',''),
                 d.get('skill_level',''), d.get('path_choice','path1')))
            conn.commit()
            log_progress(username, 'signup', d.get('standard',''))
            return jsonify({"status":"success","user":{
                "name":username,"standard":d.get('standard',''),"gpa":d.get('gpa',''),
                "goals":d.get('goals',''),"gender":d.get('gender',''),"dob":d.get('dob',''),
                "college":d.get('college',''),"department":d.get('department',''),
                "maths_grade":d.get('maths_grade',''),"cs_grade":d.get('cs_grade',''),
                "physics_grade":d.get('physics_grade',''),"english_grade":d.get('english_grade',''),
                "skill_level":d.get('skill_level',''),"path_choice":d.get('path_choice','path1'),
                "career_path":""}})
        except sqlite3.IntegrityError:
            return jsonify({"status":"error","message":"Username already exists"}), 400
        finally: conn.close()

    elif d.get('action') == 'login':
        row = c.execute('''SELECT standard,gpa,goals,gender,dob,college,department,
                            maths_grade,cs_grade,physics_grade,english_grade,
                            skill_level,path_choice,career_path
                           FROM users WHERE username=? AND password=?''',
                        (username, password)).fetchone()
        conn.close()
        if row:
            log_progress(username,'login')
            return jsonify({"status":"success","user":{
                "name":username,"standard":row['standard'],"gpa":row['gpa'],
                "goals":row['goals'] or '',"gender":row['gender'] or '',
                "dob":row['dob'] or '',"college":row['college'] or '',
                "department":row['department'] or '',
                "maths_grade":row['maths_grade'] or '',"cs_grade":row['cs_grade'] or '',
                "physics_grade":row['physics_grade'] or '',"english_grade":row['english_grade'] or '',
                "skill_level":row['skill_level'] or '',"path_choice":row['path_choice'] or 'path1',
                "career_path":row['career_path'] or ''}})
        return jsonify({"status":"error","message":"Invalid credentials"}), 401

# ══════════════════════════════════════════════════════════════
# UPDATE PROFILE
# ══════════════════════════════════════════════════════════════
@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    d = request.json; username = d.get('username','')
    try:
        conn = get_db()
        conn.execute('''UPDATE users SET
            standard=?,gpa=?,goals=?,gender=?,dob=?,college=?,department=?,
            maths_grade=?,cs_grade=?,physics_grade=?,english_grade=?,
            skill_level=?,path_choice=? WHERE username=?''',
            (d.get('standard',''), float(d.get('gpa') or 0), d.get('goals',''),
             d.get('gender',''), d.get('dob',''), d.get('college',''), d.get('department',''),
             d.get('maths_grade',''), d.get('cs_grade',''), d.get('physics_grade',''),
             d.get('english_grade',''), d.get('skill_level',''),
             d.get('path_choice','path1'), username))
        conn.commit(); conn.close()
        log_progress(username,'profile_update')
        return jsonify({"status":"success"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ══════════════════════════════════════════════════════════════
# YOUTUBE VIDEOS
# ══════════════════════════════════════════════════════════════
@app.route('/api/get_youtube_videos', methods=['POST'])
def get_youtube_videos():
    d = request.json
    skill = d.get('skill','').strip()
    role  = d.get('role','').strip()
    if not skill and not role:
        return jsonify({"videos":[],"message":"No skill or role provided"}), 400
    query  = f"{skill} tutorial beginners 2024" if skill else f"{role} full course 2024"
    videos = search_youtube_live(query, max_results=3)
    if not videos:
        videos = get_videos_for_skill(role, skill)
    for v in videos:
        if v.get('channel') == 'WhatNxt Curated' and skill:
            v['title'] = f"{skill} — Step-by-Step Tutorial"
    return jsonify({"videos":videos,"query":query,"total":len(videos)})

# ══════════════════════════════════════════════════════════════
# 🗺️ ROADMAP — RICH, DETAILED, BEAUTIFUL
# ══════════════════════════════════════════════════════════════
@app.route('/api/generate_roadmap', methods=['POST'])
def generate_roadmap():
    d    = request.json
    role = d.get('role','Software Engineer')
    name = d.get('name','Student')
    std  = d.get('standard','1st Year BTech')
    rows = get_career_rows(role)
    ctx  = get_career_context(role)

    yt_blocks = []; course_blocks = []
    for r in rows:
        skills_label = r.get('Required_Skills','').split(';')[0].strip()
        embed = r.get('Phase_1_YouTube_Embed','')
        if embed and embed != 'nan':
            if 'watch?v=' in embed:
                vid = embed.split('watch?v=')[-1].split('&')[0]
                embed = f"https://www.youtube.com/embed/{vid}"
            yt_blocks.append(
                f'<h4 style="margin:12px 0 6px;color:#60a5fa;font-size:14px;font-weight:700;">▶ {skills_label}</h4>'
                f'<iframe width="100%" height="215" src="{embed}" frameborder="0" '
                f'allowfullscreen style="border-radius:12px;margin-bottom:12px;box-shadow:0 4px 20px rgba(0,0,0,0.4);"></iframe>'
            )
        for url, tag in [(r.get('Phase_2_Course_Link',''),"💳 Paid Course"),
                         (r.get('Free_Course_1',''),"🆓 Free Course"),
                         (r.get('Free_Course_2',''),"🆓 Free Course"),
                         (r.get('Free_Course_3',''),"🆓 Free Course")]:
            if url and str(url) != 'nan':
                color = "#06b6d4" if "Paid" in tag else "#22c55e"
                course_blocks.append(
                    f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px;'
                    f'background:{color}22;color:{color};border:1px solid {color}44;'
                    f'border-radius:20px;padding:6px 14px;font-weight:600;font-size:13px;'
                    f'text-decoration:none;">{tag} — {skills_label}</a>'
                )

    # ── Build Certification Links ──
    CERT_URLS = {
        "aws cloud practitioner": "https://aws.amazon.com/certification/certified-cloud-practitioner/",
        "aws solutions architect associate": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
        "aws developer": "https://aws.amazon.com/certification/certified-developer-associate/",
        "aws ml specialty": "https://aws.amazon.com/certification/certified-machine-learning-specialty/",
        "oracle java se": "https://education.oracle.com/java-se-programmer/pexam_1Z0-829",
        "meta back-end developer": "https://www.coursera.org/professional-certificates/meta-back-end-developer",
        "meta front-end developer": "https://www.coursera.org/professional-certificates/meta-front-end-developer",
        "spring professional": "https://www.vmware.com/learning/certification/spring-professional-develop-exam.html",
        "docker certified associate": "https://training.mirantis.com/dca-certification-exam/",
        "google data analytics": "https://www.coursera.org/professional-certificates/google-data-analytics",
        "google cybersecurity": "https://www.coursera.org/professional-certificates/google-cybersecurity",
        "google ux design": "https://www.coursera.org/professional-certificates/google-ux-design",
        "google cloud ace": "https://cloud.google.com/learn/certification/cloud-engineer",
        "google professional cloud architect": "https://cloud.google.com/learn/certification/cloud-architect",
        "gcp data engineer": "https://cloud.google.com/learn/certification/data-engineer",
        "ibm data science": "https://www.coursera.org/professional-certificates/ibm-data-science",
        "coursera ml specialization": "https://www.coursera.org/specializations/machine-learning-introduction",
        "tableau desktop specialist": "https://www.tableau.com/learn/certification/desktop-specialist",
        "microsoft power bi da-100": "https://learn.microsoft.com/en-us/credentials/certifications/data-analyst-associate/",
        "comptia security+": "https://www.comptia.org/certifications/security",
        "comptia pentest+": "https://www.comptia.org/certifications/pentest",
        "comptia cysa+": "https://www.comptia.org/certifications/cybersecurity-analyst",
        "ceh": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/",
        "oscp": "https://www.offsec.com/courses/pen-200/",
        "ejpt": "https://security.ine.com/certifications/ejpt-certification/",
        "splunk core certified user": "https://www.splunk.com/en_us/training/certification-track/splunk-core-certified-user.html",
        "sans giac": "https://www.giac.org/certifications/",
        "az-900": "https://learn.microsoft.com/en-us/credentials/certifications/azure-fundamentals/",
        "az-104": "https://learn.microsoft.com/en-us/credentials/certifications/azure-administrator/",
        "az-305": "https://learn.microsoft.com/en-us/credentials/certifications/azure-solutions-architect/",
        "kubernetes cka": "https://www.cncf.io/training/certification/cka/",
        "terraform associate": "https://www.hashicorp.com/en/certification/terraform-associate",
        "tensorflow developer": "https://www.tensorflow.org/certificate",
        "deeplearning.ai tensorflow developer": "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
        "deeplearning.ai specializations": "https://www.deeplearning.ai/courses/",
        "nvidia dli": "https://www.nvidia.com/en-us/training/",
        "mongodb developer": "https://learn.mongodb.com/pages/mongodb-associate-developer-exam",
        "mlops zoomcamp": "https://github.com/DataTalksClub/mlops-zoomcamp",
        "gcp professional ml engineer": "https://cloud.google.com/learn/certification/machine-learning-engineer",
    }
    cert_blocks = []
    for r in rows:
        certs_raw = r.get('Certifications', '')
        if not certs_raw or str(certs_raw) == 'nan':
            continue
        for cert_name in str(certs_raw).split(','):
            cert_name = cert_name.strip()
            if not cert_name:
                continue
            cert_key = cert_name.lower()
            url = CERT_URLS.get(cert_key, '')
            if not url:
                for key, link in CERT_URLS.items():
                    if key in cert_key or cert_key in key:
                        url = link
                        break
            if not url:
                url = f"https://www.google.com/search?q={cert_name.replace(' ', '+')}+certification+official"
            cert_blocks.append(
                f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px;'
                f'background:#a855f722;color:#a855f7;border:1px solid #a855f744;'
                f'border-radius:20px;padding:8px 16px;font-weight:700;font-size:13px;'
                f'text-decoration:none;">🏅 {cert_name} →</a>'
            )
    cert_blocks = list(dict.fromkeys(cert_blocks))
    cert_html = "\n".join(cert_blocks) or '<p style="color:#64748b;">Check Coursera and Google for certifications.</p>'

    yt_html  = "\n".join(yt_blocks) or '<p style="color:#64748b;">No videos found for this role.</p>'
    crs_html = "\n".join(course_blocks) or '<p style="color:#64748b;">Check Coursera and Udemy for courses.</p>'
    job_title = rows[0].get('Demo_Job_Title','Software Developer') if rows else 'Software Developer'
    company   = rows[0].get('Demo_Company','Top Tech Company') if rows else 'Top Tech Company'
    salary    = rows[0].get('Salary_Range','₹6–20 LPA') if rows else '₹6–20 LPA'
    companies = rows[0].get('Top_Companies','TCS, Infosys, Wipro, Google, Amazon') if rows else 'TCS, Infosys, Google, Amazon'

    prompt = f"""You are WhatNxt AI — India's #1 AI Career Guidance Platform. You MUST generate an EXTREMELY LONG, DETAILED, STRUCTURED, COMPREHENSIVE HTML career roadmap. This is a premium product — the output should feel like a ₹50,000 career consultation report.

STUDENT PROFILE:
• Name: {name}
• Academic Year: {std}
• Target Role: {role}

CAREER INTELLIGENCE DATA:
{ctx}

SALARY RANGE: {salary}
TOP COMPANIES: {companies}
DREAM JOB: {job_title} at {company}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY OUTPUT — YOU MUST GENERATE ALL 12 SECTIONS BELOW.
DO NOT SKIP, TRUNCATE, OR SHORTEN ANY SECTION.
EACH SECTION MUST BE WRAPPED IN ITS OWN <div> WITH INLINE CSS.
MINIMUM TOTAL OUTPUT: 2500+ WORDS OF HTML CONTENT.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1: 🎯 HERO BANNER
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);border-radius:20px;padding:48px 36px;text-align:center;position:relative;overflow:hidden;border:1px solid #334155;margin-bottom:20px;">
• Role "{role}" as <h1> with gradient text (background:linear-gradient(135deg,#3b82f6,#06b6d4,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:38px;font-weight:900)
• Address student "{name}" personally with an inspiring message
• Show salary "{salary}" in a glowing accent badge
• Add a motivational subtitle like "Your personalized career blueprint starts here"
</div>

SECTION 2: 📊 KEY METRICS DASHBOARD — 4 Stat Cards
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:20px;">
Card 1: 💰 Expected Salary — {salary} — show entry, mid, senior ranges
Card 2: 🏢 Target Companies — list 4-5 from {companies}
Card 3: ⏱️ Time to Job-Ready — realistic timeline based on {std}
Card 4: 🎓 Required Certifications — count + top 2 names
Each card: background:#1e293b; border-left:4px solid [unique color]; border-radius:16px; padding:22px
</div>

SECTION 3: 🗓️ COMPLETE 4-YEAR CAREER TIMELINE (THIS IS THE MOST IMPORTANT SECTION — MUST BE VERY DETAILED)
Generate 4 separate year cards. Each year MUST contain ALL of the following:

YEAR 1 (Color: #3b82f6):
• <h3> with year number + theme name (e.g., "Year 1 — Building the Foundation")
• Q1 (Months 1-3): 4 specific tasks with technologies
• Q2 (Months 4-6): 4 specific tasks with technologies
• Q3 (Months 7-9): 3 specific tasks + 1 mini-project
• Q4 (Months 10-12): 3 specific tasks + 1 project
• 🛠️ Skills to Master: List 8 specific skills as colored badge spans
• 📁 Projects: 2-3 detailed project ideas with tech stacks
• 🏅 Certifications: 1-2 specific certifications to pursue
• ✅ Year-End Milestone: What student should be able to do by year end

YEAR 2 (Color: #06b6d4): Same structure as Year 1 but with intermediate content
YEAR 3 (Color: #22c55e): Same structure but with advanced/specialization content  
YEAR 4 (Color: #f59e0b): Same structure but with placement prep, final projects, interview content

Each year card: background:#1e293b; border-radius:16px; padding:28px; border-left:5px solid [year color]; margin-bottom:16px

SECTION 4: 🛠️ COMPREHENSIVE SKILLS MATRIX
Two-column grid layout:
LEFT COLUMN — Technical Skills (minimum 15 skills):
List specific technologies as colored badge spans grouped by category:
• Languages: (4-5 specific languages)
• Frameworks: (4-5 specific frameworks)
• Tools: (4-5 specific tools)
• Databases: (2-3 databases)
• Cloud: (2-3 cloud technologies)
Badge style: display:inline-block;margin:4px;padding:7px 16px;border-radius:20px;font-size:12px;font-weight:700;background:[color]22;color:[color];border:1px solid [color]44

RIGHT COLUMN — Soft Skills (6 skills):
Communication, Problem Solving, Teamwork, Leadership, Time Management, Critical Thinking — each as a badge

SECTION 5: 🏅 CERTIFICATIONS (MANDATORY — INCLUDE ALL THESE LINKS EXACTLY AS PROVIDED):
<h2>🏅 Recommended Certifications</h2>
COPY THESE CERTIFICATION LINKS EXACTLY — DO NOT MODIFY OR REMOVE ANY:
{cert_html}
Add a paragraph explaining WHY each certification matters specifically for {role} careers in India.
Include details about exam cost, preparation time, and validity.

SECTION 6: 📺 LEARNING RESOURCES
{yt_html}
Add context about why these videos are recommended.

SECTION 7: 📚 COURSE RECOMMENDATIONS
{crs_html}
Add a brief description of how to use these courses effectively.

SECTION 8: 🏢 TOP HIRING COMPANIES — DETAILED TABLE
<table> with dark header row (background:#1e3a5f;color:white).
MINIMUM 8 rows with these columns: Company | HQ City | Salary Range | Key Requirements | Application Process
Include companies: {companies} and add 3-4 more relevant companies.
Table style: width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden
Cell style: padding:14px 16px;border-bottom:1px solid #334155;font-size:13px

SECTION 9: 🎯 DREAM JOB PROFILE — {job_title} at {company}
• Detailed job description (4-5 sentences)
• Day-in-the-life description (3-4 bullet points of what you'd actually do)
• 8 Required Skills with ✅ checkmarks
• Expected CTC: Entry/Mid/Senior ranges in LPA
• 5 Interview Tips specific to this role
• Common Interview Questions (list 4-5 real questions asked)

SECTION 10: 📅 30-DAY QUICK START ACTION PLAN
4 weekly cards in a 2x2 grid layout:
Week 1 "Setup & Foundations": 5 specific daily tasks
Week 2 "First Project": 5 specific daily tasks
Week 3 "Deep Dive": 5 specific daily tasks
Week 4 "Review & Plan Ahead": 5 specific daily tasks
Each task should be specific and actionable (not vague like "learn coding")

SECTION 11: 📖 RECOMMENDED BOOKS & RESOURCES
List 6 must-read books for {role} career path:
Each with: Book Title (bold) | Author | One-line why it matters | Difficulty level badge

SECTION 12: 🚀 PRO TIPS FROM INDUSTRY EXPERTS
10 tips (not 8 — TEN), each with:
• Emoji icon
• Bold title
• 2-3 sentence detailed explanation
• Make tips specific to {role} in the Indian job market

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN SYSTEM (FOLLOW EXACTLY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Outer wrapper: <div style="background:#0f172a;color:#e2e8f0;padding:32px;font-family:'Segoe UI',system-ui,sans-serif;border-radius:20px;">
• Section cards: background:#1e293b; border:1px solid #334155; border-radius:16px; padding:28px; margin:20px 0
• Primary text: color:#e2e8f0
• Muted text: color:#94a3b8
• Accent colors: #3b82f6 (blue), #06b6d4 (cyan), #22c55e (green), #f59e0b (amber), #a855f7 (purple), #f43f5e (red)
• Section headings: <h2> with emoji prefix, font-size:22px, font-weight:800, margin-bottom:16px
• Sub-headings: <h3> with color matching section accent
• ALL styling must be inline CSS — NO <style> tags
• India-specific: all salaries in LPA (₹), Indian cities, Indian companies
• Skill badges as inline-block spans with colored backgrounds
• Use proper spacing between sections (margin:20px 0)

CRITICAL RULES:
1. Output RAW HTML ONLY — no markdown, no ```html, no explanations before or after
2. DO NOT truncate — generate ALL 12 sections completely
3. MINIMUM 2500 words of content
4. Every section must have substantial content — no placeholder text
5. Include the certification links from Section 5 EXACTLY as provided
6. Make it India-focused with real company names, real salary data, real certifications"""


    try:
        html = call_gemini(prompt, max_output_tokens=32768)
        html = html.replace("```html","").replace("```","").strip()
        # ── Inject certification links server-side (guaranteed to appear) ──
        if cert_html and '🏅' not in html:
            cert_section = (
                '<div style="background:#1e293b;border:1px solid #334155;border-radius:16px;'
                'padding:24px;margin:24px 0;">'
                '<h2 style="font-size:20px;font-weight:800;color:#a855f7;margin-bottom:16px;">'
                '🏅 Recommended Certifications</h2>'
                '<p style="color:#94a3b8;margin-bottom:12px;">Click to go directly to the official certification page:</p>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{cert_html}</div>'
                '<p style="color:#94a3b8;margin-top:16px;font-size:13px;">'
                '💡 Certifications boost your resume significantly in India — many top companies like TCS, Infosys, and Wipro '
                'give preference to certified candidates during placement drives.</p></div>'
            )
            if '</div>' in html:
                last_div = html.rfind('</div>')
                html = html[:last_div] + cert_section + html[last_div:]
            else:
                html += cert_section
        log_progress(name, 'roadmap_generated', role)
        return jsonify({"roadmap": html})
    except Exception as e:
        print(f"❌ Roadmap error: {e}")
        fallback = f"""<div style="background:#0f172a;color:#e2e8f0;padding:24px;font-family:'Segoe UI',Arial,sans-serif;border-radius:16px;">
          <h2 style="background:linear-gradient(90deg,#3b82f6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:28px;font-weight:900;">🗺️ Career Roadmap: {role}</h2>
          <p style="color:#f59e0b;font-size:14px;">⚠️ AI quota reached. Add a new key at aistudio.google.com/apikey. Here's your quick roadmap:</p>
          {"".join([f'<div style="border-left:4px solid {c};background:#1e293b;padding:16px;margin:12px 0;border-radius:8px;"><strong style="color:{c}">Year {i+1}:</strong><br>{t}</div>' for i,(c,t) in enumerate(zip(["#3b82f6","#06b6d4","#22c55e","#f59e0b"],["Learn Python, DSA, Git, Linux, SQL basics. Build 2 small projects. Join coding communities.","Intermediate data structures, web frameworks, first internship. Build portfolio with 3 projects.","Specialization in your chosen stack. Open source contributions. Get 1-2 certifications. Placement prep.","Final placements, mock interviews, system design. Apply to 50+ companies. Negotiate your offer well."]))])}
          <p style="color:#64748b;font-size:12px;margin-top:16px;">🔄 API quota exhausted. Run: python app.py after adding a new key.</p></div>"""
        return jsonify({"roadmap": fallback})

# ══════════════════════════════════════════════════════════════
# 📄 RESUME BUILDER — PROFESSIONAL & DETAILED
# ══════════════════════════════════════════════════════════════
@app.route('/api/build_resume', methods=['POST'])
def build_resume():
    d    = request.json
    name = d.get('name','Student')
    role = d.get('role','Software Engineer')
    ctx  = get_career_context(role)

    prompt = f"""You are India's top ATS resume specialist and career consultant. Create a COMPLETE, STUNNING, PROFESSIONAL HTML resume. It must look like a real resume a top recruiter would love.

STUDENT DETAILS:
- Full Name: {name}
- Academic Year: {d.get('standard','Final Year')}
- CGPA: {d.get('gpa','8.0')}/10
- Target Role: {role}
- College: {d.get('college','Engineering College, India')}
- Department: {d.get('department','Computer Science Engineering')}
- Career Data: {ctx[:1200]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE RESUME SECTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. HEADER (top section, clean)
   Name in large bold (28px, #1a1a1a), role title in blue (#2563eb, 16px)
   Contact row: 📱 +91-XXXXXXXXXX | 📧 {name.lower().replace(' ','.')}@gmail.com | 🔗 linkedin.com/in/{name.lower().replace(' ','-')} | 💻 github.com/{name.lower().replace(' ','')} | 📍 India

2. PROFESSIONAL SUMMARY (4-5 sentences)
   Specific to {role}. Mention CGPA {d.get('gpa')}, key skills, career goal, and a measurable achievement.
   Make it ATS-friendly with keywords for {role}.

3. EDUCATION
   Degree | College name | Graduation Year (estimated)
   CGPA: {d.get('gpa')}/10 (Distinction)
   Relevant Coursework: list 8 specific courses relevant to {role}
   Academic Achievements: 2 bullet points

4. TECHNICAL SKILLS (organized by category with colored badge spans)
   Programming Languages: (4-5 relevant to {role})
   Frameworks & Libraries: (4-5 relevant)
   Tools & Platforms: (4-5 relevant)
   Databases: (2-3 relevant)
   Core Concepts: (4-5 relevant)
   Each skill as: <span style="background:#dbeafe;color:#1d4ed8;border-radius:4px;padding:2px 8px;margin:2px;font-size:11px;font-weight:600;display:inline-block;">[SKILL]</span>

5. PROJECTS (3 detailed projects relevant to {role})
   Each project:
   - Name + tech stack (bold) + [GitHub] link
   - 3 bullet points: what it does + tech used + measurable result
   - Make them REALISTIC and IMPRESSIVE

6. INTERNSHIP / WORK EXPERIENCE (1 entry)
   Role | Company (relevant Indian company) | Duration (2-3 months)
   3 bullet points with specific actions and numbers

7. CERTIFICATIONS (4-5 real certs relevant to {role})
   Each: Cert Name | Issuing Org | Year
   
8. ACHIEVEMENTS & AWARDS (4 items)
   Mix of: hackathon, academic rank, competition, open source contribution

9. EXTRACURRICULAR ACTIVITIES (3 items)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Background: #ffffff
- Text: #1a1a1a (dark), #374151 (body)
- Accent: #2563eb (blue headings)
- Section dividers: border-bottom: 2px solid #2563eb
- Section headers: color:#2563eb, font-size:15px, font-weight:800, text-transform:uppercase, letter-spacing:1px
- Font: Arial, sans-serif
- Max-width: 794px (A4), margin:0 auto, padding:32px
- Bullet points using • character
- Each section has good spacing (margin-bottom:20px)
- Skills as inline colored badge spans
- ALL inline CSS — no style tags
- Raw HTML ONLY — no markdown"""

    try:
        html = call_gemini(prompt, max_output_tokens=6144)
        html = html.replace("```html","").replace("```","").strip()
        log_progress(name, 'resume_built', role)
        return jsonify({"resume": html})
    except Exception as e:
        return jsonify({"status":"error","message":f"AI quota reached. Add new key at aistudio.google.com/apikey"}), 503

# ══════════════════════════════════════════════════════════════
# 🔬 CERTIFICATE SCANNER — PREMIUM ANALYSIS
# ══════════════════════════════════════════════════════════════
@app.route('/api/scan_certificate', methods=['POST'])
def scan_certificate():
    try:
        d = request.json
        header, encoded = d.get('image').split(',',1)
        mime = header.split(';')[0].split(':')[1]
        img  = base64.b64decode(encoded)

        vision_prompt = """You are WhatNxt Vision AI — an expert certificate & credential analyzer.

Analyze this certificate image thoroughly. Return a JSON object with this EXACT structure (no markdown, no code fences — raw JSON only):

{
  "certificate_name": "Full certificate title as shown on the image",
  "issuing_organization": "Organization/platform that issued it (e.g., Coursera, Udemy, Google, AWS, etc.)",
  "completion_date": "Date shown on certificate or 'Not visible'",
  "credential_id": "Credential/Certificate ID if visible, or 'Not visible'",
  "skills": [
    {"name": "Skill Name", "category": "Programming|Framework|Tool|Concept|Cloud|Database|Security|AI/ML|Other", "proficiency": "Beginner|Intermediate|Advanced"},
    {"name": "Skill 2", "category": "...", "proficiency": "..."}
  ],
  "career_relevance": {
    "matching_roles": ["Role 1", "Role 2", "Role 3"],
    "industry_demand": "High|Medium|Low",
    "salary_impact": "Brief 1-sentence about how this cert impacts salary in India"
  },
  "summary": "2-3 sentence professional summary of what this certificate demonstrates",
  "next_certifications": ["Suggested next cert 1", "Suggested next cert 2", "Suggested next cert 3"]
}

Extract 6-10 specific skills. Be precise about skill names (e.g., "TensorFlow" not "Deep Learning Framework").
If you cannot read the certificate clearly, still provide your best analysis based on visible text.
Output ONLY the JSON object — no other text."""

        # Try Gemini vision first (supports image input natively)
        global _key_idx
        _reload_keys()
        total_keys = len(GEMINI_KEYS)
        result = None

        if total_keys > 0:
            for attempt in range(total_keys * 2):
                try:
                    client = get_client()
                    key_num = _key_idx % total_keys
                    print(f"🔄 Cert scan: Trying Gemini key[{key_num}]...")
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(data=img, mime_type=mime),
                            vision_prompt
                        ],
                        config=types.GenerateContentConfig(
                            max_output_tokens=2048,
                            temperature=0.3,
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                    result = resp.text.strip()
                    if result:
                        print(f"✅ Cert scan success — Gemini vision")
                        break
                except Exception as inner_e:
                    err = str(inner_e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        _key_idx += 1
                        time.sleep(2)
                    else:
                        print(f"⚠️  Cert scan Gemini error: {err[:120]}")
                        _key_idx += 1
                        time.sleep(1)

        # Parse and return structured result
        if result:
            # Clean up any markdown fences
            result = result.replace("```json","").replace("```","").strip()
            try:
                parsed = json.loads(result)
                # Extract simple skills list for backward compatibility
                skills_list = [s["name"] for s in parsed.get("skills", [])]
                parsed["skills_csv"] = ", ".join(skills_list)
                log_progress("scanner", "cert_scanned", parsed.get("certificate_name", "Unknown"))
                return jsonify({"analysis": parsed, "skills": parsed["skills_csv"]})
            except json.JSONDecodeError:
                # AI returned non-JSON, treat as simple skills list
                return jsonify({"skills": result, "analysis": None})
        else:
            raise Exception("Vision models exhausted for cert scan")
    except Exception as e:
        print(f"⚠️  Certificate scanner fallback: {e}")
        return jsonify({
            "skills": "Python, Machine Learning, Data Analysis, SQL, Statistics",
            "analysis": None,
            "note": "AI vision unavailable — showing sample skills. Try again later."
        }), 200

# ══════════════════════════════════════════════════════════════
# JOBS
# ══════════════════════════════════════════════════════════════
def load_kaggle_jobs():
    df = load_csv(DATA_DIR / 'jobs_data.csv')
    if df.empty: return df
    job_id_col = next((c for c in df.columns if c.lower() in ('job_id','id')), None)
    for fname, id_col_kw, merge_col_kw, agg_col_kw, out_col in [
        (DATA_DIR / 'companies.csv',      ('company_id','id'), ('company_id',), ('name',), 'company_name'),
        (DATA_DIR / 'job_skills.csv',     ('job_id','id'),     ('job_id',),     ('skill',), 'merged_skills'),
        (DATA_DIR / 'job_industries.csv', ('job_id','id'),     ('job_id',),     ('industry','sector'), 'industry'),
    ]:
        sub = load_csv(fname)
        if sub.empty or not job_id_col: continue
        id_col  = next((c for c in sub.columns if c.lower() in id_col_kw), None)
        val_col = next((c for c in sub.columns if any(k in c.lower() for k in agg_col_kw)), None)
        jid_col = next((c for c in df.columns  if c.lower() in merge_col_kw), None)
        if not (id_col and val_col): continue
        if 'job_skills' in str(fname) and jid_col:
            agg = (sub.groupby(id_col)[val_col]
                   .apply(lambda x: ', '.join(x.dropna().astype(str).unique()[:6]))
                   .reset_index().rename(columns={val_col: out_col}))
            df = df.merge(agg, left_on=job_id_col, right_on=id_col, how='left')
        elif jid_col:
            df = df.merge(sub[[id_col, val_col]].rename(columns={val_col: out_col}),
                          left_on=jid_col, right_on=id_col, how='left')
    return df

# Cache jobs data at startup so we don't re-read 64MB CSV on every request
df_jobs_cached = load_kaggle_jobs()
print(f"✅ Jobs cached: {len(df_jobs_cached)} rows")

@app.route('/api/get_jobs', methods=['POST'])
def get_jobs():
    d    = request.json
    role = d.get('role','All'); loc = d.get('location','All')
    try:
        df = df_jobs_cached
        if df.empty:
            return jsonify({"jobs":[],"total":0,"message":"jobs_data.csv not found"})
        title_col = next((c for c in df.columns if c.lower() in ('title','job_title','position','role')), df.columns[0])
        loc_col   = next((c for c in df.columns if c.lower() in ('location','city','place')), None)
        if role != 'All':
            df = df[df[title_col].astype(str).str.contains(role.split()[0], case=False, na=False)]
        if loc != 'All' and loc_col:
            df = df[df[loc_col].astype(str).str.contains(loc, case=False, na=False)]
        sample = (df if not df.empty else load_kaggle_jobs()).head(12).fillna("N/A")
        raw    = sample.to_dict('records')
        prompt = f"""You are an AI Job Matchmaker for Indian students. Given this raw job data:
{json.dumps(raw, default=str)[:4000]}

Select the 6 best and most relevant jobs. For each job return this exact JSON structure.
Return ONLY a valid JSON array — no markdown, no explanation, just the JSON array.

Each job object must have exactly these keys:
{{"Logo_Emoji":"relevant emoji","Role":"job title","Company":"company name","Location":"city or Remote","Type":"Full-time or Internship or Contract","Posted":"X days ago","Salary":"realistic Indian salary range in LPA or per month","Skills_Required":"comma separated 4-5 skills","Industry":"industry sector","Apply_Link":"application URL from data or empty string","Company_Careers_URL":"company careers page or empty string"}}"""
        clean = call_gemini(prompt, max_output_tokens=2000)
        clean = clean.replace("```json","").replace("```","").strip()
        jobs  = json.loads(clean)
        return jsonify({"jobs":jobs,"total":len(jobs)})
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        return jsonify({"jobs":[],"total":0,"message":"AI formatting error. Try again."}), 500
    except Exception as e:
        print(f"get_jobs error: {e}")
        return jsonify({"jobs":[],"total":0,"message":str(e)}), 500

# ══════════════════════════════════════════════════════════════
# ✉️ COVER LETTER + COLD EMAIL
# ══════════════════════════════════════════════════════════════
@app.route('/api/generate_application', methods=['POST'])
def generate_application():
    d = request.json
    prompt = f"""You are India's top campus placement career coach. Generate a complete, compelling job application.

STUDENT: {d.get('name')} | {d.get('standard')} | {d.get('college','Engineering College')} | GPA:{d.get('gpa')}/10
Skills from certificates: {d.get('student_skills','')}
Career goal: {d.get('goals','')}

TARGET JOB: {d.get('job_title')} at {d.get('company')}
Industry: {d.get('industry','')} | Required Skills: {d.get('skills_required','')}

Generate both:

1. COVER LETTER (350-400 words, formal, professional):
   • Opening hook: Mention role + company + why you're excited (1 paragraph)
   • Body 1: Academic background + CGPA + top 3 relevant skills mapped to job requirements
   • Body 2: Two specific projects/achievements that prove you can do this job (with numbers)
   • Body 3: Why THIS company specifically — mention their product or mission
   • Closing: Strong call-to-action for interview + thank you

2. COLD EMAIL (120-150 words):
   • Subject: Attention-grabbing, role-specific
   • Body: Quick intro + strongest credential + specific ask for 15-min call
   • Professional sign-off

Return ONLY valid JSON (no markdown, no extra text):
{{"cover_letter":"full letter text with newlines as \\n","cold_email_subject":"email subject line","cold_email_body":"full email body with \\n"}}"""

    try:
        result_text = call_gemini(prompt, max_output_tokens=2048)
        clean  = result_text.replace("```json","").replace("```","").strip()
        result = json.loads(clean)
        log_progress(d.get('name',''), 'cover_letter', f"{d.get('job_title')} @ {d.get('company')}")
        return jsonify({"status":"success", **result})
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', result_text, re.DOTALL)
        if m:
            try: return jsonify({"status":"success", **json.loads(m.group())})
            except: pass
        return jsonify({"status":"error","cover_letter":result_text,"cold_email_subject":"","cold_email_body":""}), 500
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 503

# ══════════════════════════════════════════════════════════════
# 🔍 DISCOVER CAREER — RICH HTML REPORT
# ══════════════════════════════════════════════════════════════
@app.route('/api/discover_career', methods=['POST'])
def discover_career():
    d = request.json
    answers = d.get('answers',[])
    marks   = d.get('marks',{})
    scores  = {"Software Engineer":0,"Data Scientist":0,"Cybersecurity Analyst":0,
               "Cloud Architect":0,"AI Engineer":0,"Product Manager":0,
               "DevOps Engineer":0,"Mobile App Developer":0}
    kw = {"Software Engineer":["puzzle","math","code","building","automate"],
          "Data Scientist":["data","pattern","analysis","statistics","excel"],
          "Cybersecurity Analyst":["security","protect","vulnerability","hacking"],
          "Cloud Architect":["cloud","server","infrastructure","deployment"],
          "AI Engineer":["ai","intelligent","learning","experiment","research"],
          "Product Manager":["people","manage","strategy","coordinate","plan"],
          "Mobile App Developer":["app","mobile","build","user","daily"],
          "DevOps Engineer":["devops","pipeline","ci","cd","automate","linux"]}
    for ans in answers:
        al = ans.lower()
        for c, kws in kw.items():
            if any(k in al for k in kws): scores[c] += 2
    if marks.get('Maths') in ['A','A+','S','9','10']: scores["Data Scientist"]+=1; scores["AI Engineer"]+=1
    if marks.get('CS')    in ['A','A+','S','9','10']: scores["Software Engineer"]+=2
    suggested = max(scores, key=scores.get)
    ctx = get_career_context(suggested)
    total_score = sum(scores.values()) or 1
    match_pct = min(98, round((scores[suggested]/total_score)*100 + 40))

    prompt = f"""You are WhatNxt AI — India's most advanced Career Discovery System. Create a STUNNING, COMPREHENSIVE, DETAILED HTML career discovery report. Make it feel EXCITING and PERSONAL.

STUDENT: {d.get('name')} | YEAR: {d.get('standard')}
QUIZ ANSWERS: {answers}
ACADEMIC MARKS: {marks}
AI CAREER MATCH: {suggested} ({match_pct}% match)
ALL SCORES: {scores}
CAREER DATABASE: {ctx[:1500]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CREATE THIS COMPLETE REPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🎉 CELEBRATION BANNER
   Big gradient banner (linear-gradient(135deg,#1e3a5f,#0f172a)).
   Large emoji + "{suggested}" in huge gradient text.
   Show "{match_pct}% Match" in a glowing badge.
   Student name "{d.get('name')}" personally addressed.
   Tagline: exciting and motivational.

2. 🧠 WHY THIS CAREER IS PERFECT FOR YOU (6-8 reasons)
   Reference their ACTUAL answers: {answers}
   Reference their ACTUAL marks: {marks}
   Each reason: bold title + 2-sentence explanation. Use ✅ emoji.
   Be PERSONAL and SPECIFIC — not generic.

3. 💪 YOUR STRENGTH SIGNALS (6 strength cards)
   Each card (background #1e293b, colored left border):
   - Strength name (bold, colored)
   - Evidence from quiz/marks
   - How it applies to {suggested}
   Colors: #3b82f6, #06b6d4, #22c55e, #f59e0b, #a855f7, #f43f5e

4. 💰 CAREER OUTLOOK IN INDIA
   - Starting salary: specific LPA range
   - 3-year salary: specific LPA range
   - 5-year salary: specific LPA range
   - Top 8 companies hiring (real names)
   - Best 5 cities in India for this role
   - Job growth: specific percentage
   All in a styled info box.

5. 🗺️ 4-YEAR ROADMAP PREVIEW
   4 colorful timeline cards with Year 1-4. Each with:
   - 4-5 specific skills to learn
   - Key milestone to achieve
   Year colors: #3b82f6, #06b6d4, #22c55e, #f59e0b

6. 🎯 YOUR FIRST 3 ACTIONS THIS WEEK
   Very specific, can-do-today actions:
   Step 1: (specific platform + resource name)
   Step 2: (specific action with tool name)
   Step 3: (specific community or course to join)
   Each in a styled card with big number.

7. 📚 TOP FREE RESOURCES TO START NOW
   5 specific resources with names and URLs (YouTube channels, websites, platforms)
   As styled link buttons.

8. 🏆 FAMOUS INDIANS WHO MADE IT IN {suggested.upper()}
   3 real successful Indians in this field. Brief inspiring story (2 sentences each).

9. ⚠️ HONEST CHALLENGES (4 items)
   Real challenges + how to overcome them. Be honest but encouraging.

10. 🔥 PERSONAL MOTIVATIONAL MESSAGE
    Address {d.get('name')} by name. Reference their specific marks and quiz answers.
    3 paragraphs. Make them feel they CAN do this. End with a powerful quote.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN: background #0f172a, cards #1e293b, text #e2e8f0, rich inline CSS.
MINIMUM 800 words of content. Raw HTML ONLY — no markdown."""

    try:
        html = call_gemini(prompt, max_output_tokens=6144)
        html = html.replace("```html","").replace("```","").strip()
    except Exception as e:
        html = f'<div style="padding:24px;background:#1e293b;color:#e2e8f0;border-radius:16px;font-family:Arial,sans-serif;"><h2 style="color:#60a5fa;font-size:24px;">🎯 Your Best Career Match: {suggested}</h2><div style="background:#0f172a;border-left:4px solid #3b82f6;padding:16px;border-radius:8px;margin:12px 0;"><p style="margin:0;">Based on your quiz answers and academic performance, <strong style="color:#60a5fa;">{suggested}</strong> is your ideal career path with an estimated <strong style="color:#22c55e;">{match_pct}% compatibility score</strong>.</p></div><p style="color:#f59e0b;">⚠️ Full report unavailable — API quota reached. Add a new key at aistudio.google.com/apikey for the detailed report.</p></div>'

    try:
        conn = get_db()
        conn.execute("UPDATE users SET career_path=? WHERE username=?", (suggested, d.get('name')))
        conn.commit(); conn.close()
    except: pass
    log_progress(d.get('name',''), 'career_discovered', suggested)
    return jsonify({"career":suggested,"report":html,"score":scores,"match_pct":match_pct})

# ══════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════
@app.route('/api/get_quiz', methods=['POST'])
def get_quiz():
    d = request.json
    domain = d.get('domain','Software Engineer')
    count  = int(d.get('count',5))
    if df_quiz.empty: return jsonify({"error":"Quiz data not loaded"}), 500
    dq = df_quiz[df_quiz['Domain']==domain]
    if dq.empty: dq = df_quiz[df_quiz['Domain']!='Psychometric']
    sample = dq.sample(min(count,len(dq)))
    return jsonify({"questions":[{
        "question":r['Question'],
        "options":{"A":r['Option_A'],"B":r['Option_B'],"C":r['Option_C'],"D":r['Option_D']},
        "correct":r['Correct'],"explanation":r['Explanation']
    } for _,r in sample.iterrows()],"domain":domain,"total":len(sample)})

@app.route('/api/submit_quiz', methods=['POST'])
def submit_quiz():
    d = request.json
    answers = d.get('answers',{}); correct = d.get('correct_answers',{})
    score = sum(1 for i,a in answers.items() if correct.get(str(i))==a)
    total = len(correct); pct = round((score/total)*100) if total else 0
    grade = "Excellent 🏆" if pct>=80 else "Good 👍" if pct>=60 else "Average 📚" if pct>=40 else "Needs Work 💪"
    nxt   = ("Ready for advanced topics!" if pct>=80 else "Review missed topics & retry." if pct>=60
             else "Spend more time on fundamentals." if pct>=40 else "Start from basics — check Courses tab.")
    log_progress(d.get('username',''),'quiz_completed',f"{d.get('domain')}:{score}/{total}")
    return jsonify({"score":score,"total":total,"percentage":pct,"grade":grade,"next_step":nxt})

@app.route('/api/get_psychometric_quiz', methods=['GET'])
def get_psychometric_quiz():
    if df_quiz.empty: return jsonify({"error":"No data"}), 500
    pq = df_quiz[df_quiz['Domain']=='Psychometric']
    return jsonify({"questions":[{"question":r['Question'],
        "options":{"A":r['Option_A'],"B":r['Option_B'],"C":r['Option_C'],"D":r['Option_D']},
        "explanation":r['Explanation']} for _,r in pq.iterrows()]})

# ══════════════════════════════════════════════════════════════
# COURSES
# ══════════════════════════════════════════════════════════════
@app.route('/api/get_courses', methods=['POST'])
def get_courses():
    d = request.json
    domain = d.get('domain','Software Engineer')
    level  = d.get('level','All')
    if df_courses.empty: return jsonify({"courses":[]})
    f = df_courses[df_courses['Domain']==domain]
    if level and level!='All': f = f[f['Level'].str.contains(level,case=False,na=False)]
    return jsonify({"courses":f.to_dict(orient='records')})

# ══════════════════════════════════════════════════════════════
# 💬 CHAT — DETAILED, INDIA-SPECIFIC
# ══════════════════════════════════════════════════════════════
@app.route('/api/chat', methods=['POST'])
def chat():
    d      = request.json
    msg    = d.get('message','')
    name   = d.get('name','')
    career = d.get('career','')
    ctx    = get_career_context(career) if career else ""
    hist   = "".join(f"{h.get('sender','')}: {h.get('text','')}\n" for h in d.get('history',[])[-6:])

    prompt = f"""You are NxtBot, WhatNxt's expert AI Career Mentor for Indian engineering students. You are warm, encouraging, highly knowledgeable, and extremely specific in your advice. You speak from experience and care deeply about each student's success.

STUDENT PROFILE:
- Name: {name}
- Career Goal: {career or 'exploring options'}
- Career Database Info: {ctx[:800] if ctx else 'General guidance mode'}
- Recent Chat History: {hist if hist else 'First message'}

STUDENT'S MESSAGE: "{msg}"

YOUR RESPONSE MUST:
✅ Be detailed and helpful (8-12 sentences)
✅ Be SPECIFIC — name real tools, platforms, courses (e.g. "LeetCode", "GeeksforGeeks", "Coursera Google Data Analytics")
✅ Use India context — Indian companies (TCS, Infosys, Flipkart, Zomato), salaries in LPA, Indian cities
✅ If they ask about skills → list specific skills + where to learn each one
✅ If they ask about career → give a concrete 3-step plan with timeline
✅ If they ask about salary → give realistic Indian ranges at different levels
✅ If they seem stressed/confused → be extra warm and break things into tiny steps
✅ Use 2-3 emojis naturally (not too many)
✅ End with either an encouraging line OR a follow-up question
✅ Address them by name "{name}" at least once
✅ NEVER give vague, generic advice — always be specific and actionable"""

    try:
        reply = call_gemini(prompt, max_output_tokens=1000)
        log_progress(name, 'chat', msg[:50])
        return jsonify({"reply": reply.strip()})
    except Exception as e:
        return jsonify({"reply": f"Hey {name}! 😊 I'm a little busy right now but I'll be back soon. In the meantime, check out your Roadmap tab for detailed career guidance! Try again in a minute."}), 200

# ══════════════════════════════════════════════════════════════
# HEALTH / PROGRESS / YEARLY PLAN
# ══════════════════════════════════════════════════════════════
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "ai_provider": "Google Gemini",
        "models": MODELS,
        "keys_configured": len(GEMINI_KEYS),
        "datasets": {
            "career_data": len(df_career),
            "quiz": len(df_quiz),
            "courses": len(df_courses)
        }
    })

@app.route('/api/get_progress', methods=['POST'])
def get_progress():
    u = request.json.get('username','')
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT action,detail,timestamp FROM progress WHERE username=? ORDER BY timestamp DESC LIMIT 20",
            (u,)).fetchall()
        conn.close()
        return jsonify({"progress":[{"action":r[0],"detail":r[1],"time":r[2]} for r in rows]})
    except Exception as e:
        return jsonify({"progress":[],"error":str(e)})

@app.route('/api/yearly_plan', methods=['POST'])
def yearly_plan():
    d    = request.json
    role = d.get('role','Software Engineer')
    name = d.get('name','Student')
    std  = d.get('standard','1st Year BTech')
    gpa  = d.get('gpa','8.0')
    college    = d.get('college','')
    department = d.get('department','')
    skill_level = d.get('skill_level','beginner')
    goals = d.get('goals','')
    cur  = next((i for i,y in enumerate(["1st","2nd","3rd","4th"],1) if y in std), 1)
    rows = get_career_rows(role)
    ctx  = get_career_context(role)

    # CSV-based basic plan data for context
    csv_plans = {}
    if rows:
        for col, num in [("Year_1_Plan",1),("Year_2_Plan",2),("Year_3_Plan",3),("Year_4_Plan",4)]:
            if col in rows[0]: csv_plans[f"Year {num}"] = rows[0][col]

    salary    = rows[0].get('Salary_Range','₹6–20 LPA') if rows else '₹6–20 LPA'
    companies = rows[0].get('Top_Companies','TCS, Infosys, Wipro, Google, Amazon') if rows else 'TCS, Infosys, Google, Amazon'
    certs     = rows[0].get('Certifications','') if rows else ''

    prompt = f"""You are WhatNxt AI — India's #1 AI Career Planning System. Generate an EXTREMELY DETAILED, STRUCTURED, COMPREHENSIVE HTML yearly career plan. This should feel like a ₹50,000 premium career consultation document.

STUDENT PROFILE:
• Name: {name}
• Academic Year: {std} (Currently in Year {cur} of 4)
• GPA: {gpa}/10
• College: {college or 'Engineering College, India'}
• Department: {department or 'Computer Science'}
• Skill Level: {skill_level}
• Career Goal: {goals or role}
• Target Role: {role}

CAREER INTELLIGENCE DATA:
{ctx[:2000]}

CSV YEAR PLANS (use as base, but EXPAND massively):
{json.dumps(csv_plans, default=str)}

SALARY RANGE: {salary}
TOP COMPANIES: {companies}
CERTIFICATIONS: {certs}
CURRENT YEAR: {cur}
REMAINING YEARS: {list(range(cur,5))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY OUTPUT — YOU MUST GENERATE ALL SECTIONS BELOW.
DO NOT SKIP, TRUNCATE, OR SHORTEN ANY SECTION.
MINIMUM TOTAL OUTPUT: 3000+ WORDS OF HTML CONTENT.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1: 🎯 PERSONALIZED PLAN HEADER
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);border-radius:20px;padding:48px 36px;text-align:center;position:relative;overflow:hidden;border:1px solid #334155;margin-bottom:24px;">
• "Your {4-cur+1}-Year Career Blueprint" as <h1> with gradient text
• Address "{name}" personally
• Show current position: "Year {cur} of 4 · {std}"
• Target: "{role}" with salary "{salary}" in a glowing badge
• A motivational line about the journey ahead
</div>

SECTION 2: 📊 CURRENT STATUS ASSESSMENT
<div> with 4 stat cards in a grid:
Card 1: 📍 Current Position — Year {cur}, {std}, GPA {gpa}
Card 2: 🎯 Target Role — {role}
Card 3: 💰 Salary Potential — {salary} with growth trajectory
Card 4: ⏱️ Time Remaining — {4-cur} years to job-ready
Each card: background:#1e293b; border-left:4px solid [unique color]; border-radius:16px; padding:22px
</div>

SECTION 3: 🗓️ DETAILED YEAR-BY-YEAR PLAN (MOST IMPORTANT — EXTREMELY DETAILED)
Generate a detailed plan for EACH remaining year from Year {cur} to Year 4.
Each year MUST be a separate card with ALL of the following sub-sections:

For EACH year (color cycle: #3b82f6, #06b6d4, #22c55e, #f59e0b):
<div style="background:#1e293b;border-radius:16px;padding:28px;border-left:5px solid [year-color];margin-bottom:20px;">

  <h3> Year N — "[Theme Name]" (e.g., "Building the Foundation", "Deep Specialization", etc.)</h3>

  📅 SEMESTER 1 (Months 1-6):
  • Month 1-2: 4 specific tasks with exact technologies, tools, platforms
  • Month 3-4: 4 specific tasks building on previous months
  • Month 5-6: 4 specific tasks + 1 mini-project with full tech stack
  Each task must name SPECIFIC technologies (e.g., "Complete Python for Everybody on Coursera", not "learn coding")

  📅 SEMESTER 2 (Months 7-12):
  • Month 7-8: 4 specific tasks
  • Month 9-10: 4 specific tasks + 1 intermediate project
  • Month 11-12: 3 tasks + 1 major project + portfolio update

  🛠️ SKILLS TO MASTER THIS YEAR:
  List 10-12 specific skills as colored badge spans grouped by:
  - Programming Languages (3-4)
  - Frameworks & Libraries (3-4)
  - Tools & Platforms (3-4)
  - Soft Skills (2-3)
  Badge style: display:inline-block;margin:4px;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;background:[color]22;color:[color];border:1px solid [color]44

  📁 PROJECTS TO BUILD (3-4 projects):
  Each project must include:
  - Project name (bold)
  - Full tech stack
  - 2-3 sentence description of what it does
  - Expected outcome (e.g., "Deploy to Heroku, get 50+ GitHub stars")
  - Difficulty level badge

  🏅 CERTIFICATIONS TO PURSUE (2-3 per year):
  - Certification name + issuing body
  - Estimated prep time
  - Exam cost
  - Why it matters for {role}

  📚 COURSES & RESOURCES (4-6 per year):
  - Course name + platform + free/paid
  - Estimated hours
  - Specific URL if available

  🎯 INTERNSHIP / EXPERIENCE GOALS:
  - What type of internship to target
  - Companies to apply to (name 5-6 specific companies)
  - How to prepare for interviews
  - Portfolio items needed

  ✅ YEAR-END MILESTONES (5-6 checkpoints):
  What student should be able to do/have by year end
  Each as a checkbox-style item with ✅

  📈 MONTHLY SCHEDULE TABLE:
  A <table> with columns: Month | Focus Area | Key Deliverable | Hours/Week
  12 rows (one per month)
  Table style: width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden
</div>

SECTION 4: 🎯 PLACEMENT PREPARATION TIMELINE
<div> Detailed placement prep plan:
- When to start DSA practice (which month, which platform — LeetCode, GeeksforGeeks, CodeForces)
- Mock interview schedule (weekly/monthly breakdown)
- Company-specific preparation (for {companies})
- Resume building milestones
- LinkedIn optimization steps
- GitHub portfolio requirements (minimum repos, stars, contributions)
- Competitive programming targets
</div>

SECTION 5: 📊 WEEKLY ROUTINE TEMPLATE
<div> A model weekly schedule for the current year ({std}):
A visual weekly planner table:
| Day | Morning (2hrs) | Afternoon (2hrs) | Evening (1hr) |
Fill with specific activities like "DSA on LeetCode", "Project work", "Course videos", "Mock interviews"
</div>

SECTION 6: 🏢 TARGET COMPANIES BY YEAR
<table> with columns: Company | When to Apply | Required Skills | Package Range | Application Process
Minimum 10 companies including {companies} + startups + MNCs
Group by: Dream (Year 4), Target (Year 3-4), Safe (Year 2-3)
</table>

SECTION 7: 💡 SUCCESS STRATEGIES & PRO TIPS
10 detailed, actionable tips specific to {role} career in India:
Each with emoji + bold title + 3-4 sentence explanation
Include: networking strategies, GitHub optimization, LinkedIn tips, interview prep, salary negotiation

SECTION 8: 🚀 30-DAY IMMEDIATE ACTION PLAN
4 weekly cards (Week 1-4) in a 2x2 grid:
Each week has 7 daily tasks (one per day, Monday-Sunday)
Tasks must be ultra-specific: "Day 1: Create GitHub account, set up profile, push first repo"
Not vague like "start learning"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN SYSTEM (FOLLOW EXACTLY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Outer wrapper: <div style="background:#0f172a;color:#e2e8f0;padding:32px;font-family:'Segoe UI',system-ui,sans-serif;border-radius:20px;">
• Section cards: background:#1e293b; border:1px solid #334155; border-radius:16px; padding:28px; margin:20px 0
• Primary text: color:#e2e8f0
• Muted text: color:#94a3b8
• Accent colors: #3b82f6 (blue), #06b6d4 (cyan), #22c55e (green), #f59e0b (amber), #a855f7 (purple), #f43f5e (red)
• Section headings: <h2> with emoji prefix, font-size:22px, font-weight:800, margin-bottom:16px
• Year cards: border-left:5px solid [year-color]
• ALL styling must be inline CSS — NO <style> tags
• India-specific: salaries in LPA (₹), Indian cities, Indian companies
• Skill badges as inline-block spans with colored backgrounds
• Tables with dark header (background:#1e3a5f;color:white)

CRITICAL RULES:
1. Output RAW HTML ONLY — no markdown, no ```html, no explanations
2. DO NOT truncate — generate ALL sections completely
3. MINIMUM 3000 words of content
4. Every section must have substantial, detailed content
5. Be India-focused: real companies, real salaries, real platforms
6. Make it feel like a premium ₹50,000 career consultation report
7. Address "{name}" by name throughout the document
8. Be SPECIFIC — name exact tools, platforms, courses, companies"""

    try:
        html = call_gemini(prompt, max_output_tokens=32768)
        html = html.replace("```html","").replace("```","").strip()
        log_progress(name, 'yearly_plan_generated', f"{role} Year {cur}")
        return jsonify({
            "current_year": cur,
            "role": role,
            "plans": csv_plans,
            "remaining_years": list(range(cur,5)),
            "detailed_plan": html
        })
    except Exception as e:
        print(f"❌ Yearly plan AI error: {e}")
        # Fallback to basic CSV data if AI fails
        return jsonify({
            "current_year": cur,
            "role": role,
            "plans": csv_plans,
            "remaining_years": list(range(cur,5)),
            "detailed_plan": None,
            "error": "AI quota exhausted — showing basic plan. Add a new key at aistudio.google.com/apikey"
        })

if __name__ == '__main__':
    print(f"🚀 WhatNxt API — http://127.0.0.1:5001")
    print(f"🤖 AI Primary   : Gemini ({', '.join(GEMINI_MODELS)})")
    print(f"🔑 Gemini Keys  : {len(GEMINI_KEYS)} (PRIMARY)")
    print(f"🌐 OpenRouter   : {'✅ Enabled (FALLBACK — ' + OPENROUTER_MODEL + ')' if OPENROUTER_API_KEY else '❌ Disabled'}")
    print(f"📊 Career rows  : {len(df_career)}")
    print(f"📝 Quiz rows    : {len(df_quiz)}")
    print(f"📚 Course rows  : {len(df_courses)}")
    print(f"💡 Gemini keys  : https://aistudio.google.com/apikey")
    print(f"💡 OpenRouter   : https://openrouter.ai/keys")
    app.run(port=5001, debug=True, use_reloader=False)