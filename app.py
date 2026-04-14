from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import sqlite3, pandas as pd, base64, json, re, os, time
from pathlib import Path

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
# 🔑 API KEYS — loaded from .env file (never hardcode secrets!)
# Get free keys at: https://aistudio.google.com/apikey
# ═══════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')

GEMINI_KEYS = [k.strip() for k in os.environ.get('GEMINI_KEYS', '').split(',') if k.strip()]
if not GEMINI_KEYS:
    print("⚠️  No GEMINI_KEYS found! Copy .env.example to .env and add your keys.")
    print("   Get free keys at: https://aistudio.google.com/apikey")

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', '')
_key_idx = 0

def get_client():
    return genai.Client(api_key=GEMINI_KEYS[_key_idx % len(GEMINI_KEYS)])

# ── Models in priority order ─────────────────────────────────
MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# ═══════════════════════════════════════════════════════════════
# 🤖 SMART AI CALLER — retries + key rotation + model fallback
# ═══════════════════════════════════════════════════════════════
def call_gemini(prompt, max_output_tokens=8192):
    global _key_idx
    for model in MODELS:
        for attempt in range(2):
            try:
                client = get_client()
                print(f"🔄 Trying {model} with key[{_key_idx % len(GEMINI_KEYS)}]...")
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=0.8,
                    )
                )
                print(f"✅ Success — {model}")
                return resp.text
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    print(f"⚠️  Key[{_key_idx % len(GEMINI_KEYS)}] quota hit — rotating key...")
                    _key_idx += 1
                    time.sleep(1)
                elif "503" in err or "UNAVAILABLE" in err:
                    print(f"⚠️  {model} busy, retrying in 3s...")
                    time.sleep(3)
                elif "404" in err or "NOT_FOUND" in err:
                    print(f"⚠️  {model} not found, trying next...")
                    break
                else:
                    print(f"⚠️  {model}: {err[:120]}")
                    time.sleep(2)
    raise Exception("All Gemini models exhausted. Add a new API key at https://aistudio.google.com/apikey")

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

    yt_html  = "\n".join(yt_blocks) or '<p style="color:#64748b;">No videos found for this role.</p>'
    crs_html = "\n".join(course_blocks) or '<p style="color:#64748b;">Check Coursera and Udemy for courses.</p>'
    job_title = rows[0].get('Demo_Job_Title','Software Developer') if rows else 'Software Developer'
    company   = rows[0].get('Demo_Company','Top Tech Company') if rows else 'Top Tech Company'
    salary    = rows[0].get('Salary_Range','₹6–20 LPA') if rows else '₹6–20 LPA'
    companies = rows[0].get('Top_Companies','TCS, Infosys, Wipro, Google, Amazon') if rows else 'TCS, Infosys, Google, Amazon'

    prompt = f"""You are WhatNxt AI — India's most advanced Career Guidance System. Generate an EXTREMELY DETAILED, BEAUTIFUL, COMPREHENSIVE HTML career roadmap page. This must look STUNNING and be packed with useful, India-specific information.

STUDENT: {name} | YEAR: {std} | TARGET ROLE: {role}
CAREER DATA FROM DATABASE: {ctx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY SECTIONS — MAKE EACH ONE RICH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🎯 HERO BANNER
   Large dark banner (background: linear-gradient(135deg,#0f172a,#1e3a5f)). 
   Show: student name "{name}", role "{role}", motivational tagline, salary "{salary}".
   Big colorful gradient text for the role name.

2. 📊 4 STAT CARDS IN A ROW
   - 💰 Starting Salary: {salary}
   - 🏢 Top Companies: first 3 from "{companies}"
   - ⏱️ Time to Job Ready: calculate based on year {std}
   - 🎓 Must-Have Certifications: pick 3 from career data
   Cards: background #1e293b, colorful left borders, rounded corners.

3. 🗓️ YEAR-BY-YEAR ROADMAP CARDS
   Create a beautiful timeline card for EACH year from "{std}" to Year 4.
   Each card MUST have:
   ✅ 6-8 SPECIFIC skills (not generic — e.g. "Python Pandas for data wrangling" not "Python")
   ✅ 2-3 project ideas with brief descriptions
   ✅ Certifications to earn
   ✅ Quarterly breakdown (Q1/Q2/Q3/Q4 goals)
   ✅ End-of-year target (internship / job / promotion)
   Card colors: Year1=border-left:#3b82f6, Year2=#06b6d4, Year3=#22c55e, Year4=#f59e0b
   Card background: #1e293b, padding 20px, border-radius 12px, margin-bottom 16px

4. 🛠️ SKILLS MATRIX — Two styled columns:
   LEFT: Technical Skills (12+ skills as colorful badge spans — background color chips)
   RIGHT: Soft Skills (6 skills as badge spans)
   
5. 📺 LEARNING RESOURCES
   Section title with 📺 emoji. Show these embedded videos:
   {yt_html}

6. 📚 COURSES & RESOURCES
   Section title. Show all these links as styled pill buttons:
   {crs_html}

7. 🏢 TOP COMPANIES HIRING TABLE
   Styled HTML table (dark header #1e3a5f) with 8 companies:
   Columns: Company | City | Salary Range | What They Look For | How to Apply
   Use real companies from: {companies} plus others relevant to {role}

8. 🎯 DREAM JOB SPOTLIGHT
   Highlighted box (gradient border): {job_title} at {company}
   Include: role description (3 sentences), required skills checklist (6 items with ✅),
   interview tips (5 specific tips), salary negotiation advice.

9. 📅 FIRST 30 DAYS ACTION PLAN
   4 weekly cards (Week 1, 2, 3, 4) each with 4-5 specific daily tasks.
   Card background #0f172a, colored top border, emoji per task.

10. 🚀 PRO TIPS FOR {role.upper()} IN INDIA
    8 specific, actionable tips. Each with emoji, bold title, 2-sentence explanation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT DESIGN RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Overall background: #0f172a (dark navy)
- Card background: #1e293b
- Card border: 1px solid #334155
- Primary text: #e2e8f0
- Muted text: #94a3b8
- Accent colors: #3b82f6 (blue) #06b6d4 (cyan) #22c55e (green) #f59e0b (amber) #a855f7 (purple) #f43f5e (red)
- Font: font-family:'Segoe UI',system-ui,Arial,sans-serif; font-size:14px; line-height:1.6
- Wrap everything in: <div style="background:#0f172a;color:#e2e8f0;padding:24px;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;">
- ALL styling must be inline CSS — no <style> tags, no external CSS
- Section headers: font-size:20px, font-weight:800, margin-bottom:16px, with colored emoji
- Generous padding and spacing — make it breathe
- Use emojis liberally to make it visual and engaging
- India-specific: all salaries in LPA, mention Indian cities, Indian companies
- MINIMUM 1200 words of actual content
- Output RAW HTML ONLY — absolutely no markdown, no ```html fences, no explanations"""

    try:
        html = call_gemini(prompt, max_output_tokens=8192)
        html = html.replace("```html","").replace("```","").strip()
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
# CERTIFICATE SCANNER
# ══════════════════════════════════════════════════════════════
@app.route('/api/scan_certificate', methods=['POST'])
def scan_certificate():
    try:
        d = request.json
        header, encoded = d.get('image').split(',',1)
        mime = header.split(';')[0].split(':')[1]
        img  = base64.b64decode(encoded)
        client = get_client()
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Part.from_bytes(data=img, mime_type=mime),
                "You are a skill extractor. Look at this certificate image carefully. Extract the top 4-6 technical skills, tools, or technologies this certificate covers. Return ONLY a comma-separated list of skills. Example: Python, Machine Learning, TensorFlow, Data Analysis, Neural Networks"
            ]
        )
        return jsonify({"skills": resp.text.strip()})
    except Exception as e:
        return jsonify({"skills":"Python, Machine Learning, Data Analysis, SQL, Statistics","note":"Manual skill entry mode"}), 200

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

@app.route('/api/get_jobs', methods=['POST'])
def get_jobs():
    d    = request.json
    role = d.get('role','All'); loc = d.get('location','All')
    try:
        df = load_kaggle_jobs()
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
    d   = request.json
    role = d.get('role','Software Engineer')
    std  = d.get('standard','1st Year BTech')
    cur  = next((i for i,y in enumerate(["1st","2nd","3rd","4th"],1) if y in std), 1)
    rows = get_career_rows(role)
    plans = {}
    if rows:
        for col, num in [("Year_1_Plan",1),("Year_2_Plan",2),("Year_3_Plan",3),("Year_4_Plan",4)]:
            if col in rows[0]: plans[f"Year {num}"] = rows[0][col]
    return jsonify({"current_year":cur,"role":role,"plans":plans,"remaining_years":list(range(cur,5))})

if __name__ == '__main__':
    print(f"🚀 WhatNxt API — http://127.0.0.1:5000")
    print(f"🤖 AI Provider  : Google Gemini (gemini-1.5-flash primary)")
    print(f"🔑 Keys loaded  : {len(GEMINI_KEYS)}")
    print(f"📊 Career rows  : {len(df_career)}")
    print(f"📝 Quiz rows    : {len(df_quiz)}")
    print(f"📚 Course rows  : {len(df_courses)}")
    print(f"💡 To add more keys: edit GEMINI_KEYS list at top of file")
    print(f"🆓 Get free key : https://aistudio.google.com/apikey")
    app.run(port=5000, debug=True, use_reloader=False)