from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import sqlite3
import pandas as pd
import base64
import random
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Gemini Client ─────────────────────────────────────────────
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
MODEL = "gemini-2.5-flash"

# ─── Load All Datasets ─────────────────────────────────────────
def load_csv(name):
    try:
        return pd.read_csv(f'../data/{name}')
    except Exception as e:
        print(f"Warning: Could not load {name}: {e}")
        return pd.DataFrame()

df_career   = load_csv('career_data(Claude).csv')
df_jobs     = load_csv('jobs_data.csv')
df_quiz     = load_csv('quiz_data.csv')
df_courses  = load_csv('courses_data(Claude).csv')
df_psycho   = load_csv('psychometric_map.csv')

# ─── Database ─────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('whatnxt.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT UNIQUE, password TEXT, standard TEXT,
        gpa REAL, goals TEXT, career_path TEXT, quiz_scores TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS progress (
        username TEXT, action TEXT, detail TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()
init_db()

def log_progress(username, action, detail=""):
    try:
        conn = sqlite3.connect('whatnxt.db')
        conn.execute("INSERT INTO progress (username, action, detail) VALUES (?,?,?)", (username, action, detail))
        conn.commit()
        conn.close()
    except: pass

# ─── Helper: CSV context injection for RAG ─────────────────────
def get_career_context(role):
    rows = df_career[df_career['Target_Role'] == role] if not df_career.empty else pd.DataFrame()
    if rows.empty:
        return ""
    ctx = []
    for _, r in rows.iterrows():
        ctx.append(f"""
Role: {r.get('Target_Role','')}
Required Skills: {r.get('Required_Skills','')}
Year 1 Plan: {r.get('Year_1_Plan','')}
Year 2 Plan: {r.get('Year_2_Plan','')}
Year 3 Plan: {r.get('Year_3_Plan','')}
Year 4 Plan: {r.get('Year_4_Plan','')}
Top Companies: {r.get('Top_Companies','')}
Salary Range: {r.get('Salary_Range','')}
Certifications: {r.get('Certifications','')}
Demo Job: {r.get('Demo_Job_Title','')} at {r.get('Demo_Company','')}
Course Link: {r.get('Phase_2_Course_Link','')}
YouTube: {r.get('Phase_1_YouTube_Embed','')}
        """)
    return "\n".join(ctx)

def get_jobs_context(role):
    if df_jobs.empty: return ""
    # Match role broadly
    mask = df_jobs['Role'].str.contains(role.split()[0], case=False, na=False)
    rows = df_jobs[mask].head(5)
    if rows.empty:
        rows = df_jobs.head(5)
    lines = []
    for _, r in rows.iterrows():
        lines.append(f"{r['Logo_Emoji']} {r['Role']} @ {r['Company']} | {r['Location']} | {r['Salary']} | {r['Type']} | Skills: {r['Skills_Required']} | Apply: {r['Apply_Link']}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════
@app.route('/api/auth', methods=['POST'])
def auth():
    data = request.json
    action, username, password = data.get('action'), data.get('username'), data.get('password')
    conn = sqlite3.connect('whatnxt.db')
    c = conn.cursor()
    if action == 'signup':
        try:
            c.execute('INSERT INTO users (username,password,standard,gpa,goals) VALUES (?,?,?,?,?)',
                      (username, password, data.get('standard'), float(data.get('gpa',0)), data.get('goals','')))
            conn.commit()
            log_progress(username, 'signup', data.get('standard'))
            return jsonify({"status":"success","user":{"name":username,"standard":data.get('standard'),"gpa":data.get('gpa'),"goals":data.get('goals','')}})
        except Exception as e:
            return jsonify({"status":"error","message":"Username already exists"}), 400
        finally: conn.close()
    elif action == 'login':
        c.execute('SELECT standard,gpa,goals,career_path FROM users WHERE username=? AND password=?', (username,password))
        user = c.fetchone()
        conn.close()
        if user:
            log_progress(username, 'login')
            return jsonify({"status":"success","user":{"name":username,"standard":user[0],"gpa":user[1],"goals":user[2] or "","career_path":user[3] or ""}})
        return jsonify({"status":"error","message":"Invalid credentials"}), 401

# ═══════════════════════════════════════════════════════════════
# UPDATE PROFILE
# ═══════════════════════════════════════════════════════════════
@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    data = request.json
    username = data.get('username')
    try:
        conn = sqlite3.connect('whatnxt.db')
        conn.execute('UPDATE users SET gpa=?, goals=?, standard=? WHERE username=?',
                     (float(data.get('gpa', 0)), data.get('goals',''), data.get('standard',''), username))
        conn.commit()
        conn.close()
        log_progress(username, 'profile_update')
        return jsonify({"status":"success"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ═══════════════════════════════════════════════════════════════
# CERTIFICATE SCANNER (Vision AI)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/scan_certificate', methods=['POST'])
def scan_certificate():
    data = request.json
    header, encoded = data.get('image').split(',', 1)
    mime_type = header.split(';')[0].split(':')[1]
    image_bytes = base64.b64decode(encoded)
    prompt = """Analyze this educational certificate or document.
Extract ONLY the top 3-5 core technical skills or subjects learned.
Return them as a clean comma-separated list only. No explanations, no bullet points."""
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
    )
    return jsonify({"skills": response.text.strip()})

# ═══════════════════════════════════════════════════════════════
# ROADMAP GENERATOR (RAG-enhanced)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/generate_roadmap', methods=['POST'])
def generate_roadmap():
    data = request.json
    role = data.get('role', 'Software Engineer')
    standard = data.get('standard', '1st Year BTech')
    name = data.get('name', 'Student')

    # Get CSV context for this role
    career_ctx = get_career_context(role)

    # Find YouTube embed and course link from CSV
    role_rows = df_career[df_career['Target_Role'] == role] if not df_career.empty else pd.DataFrame()
    yt_embed = ""
    course_link = ""
    job_title = ""
    company = ""
    if not role_rows.empty:
        row = role_rows.iloc[0]
        yt_embed = row.get('Phase_1_YouTube_Embed', '')
        course_link = row.get('Phase_2_Course_Link', '')
        job_title = row.get('Demo_Job_Title', '')
        company = row.get('Demo_Company', '')

    youtube_block = ""
    if yt_embed:
        youtube_block = f"""
CRITICAL: Embed this exact YouTube video using this iframe (do not modify it):
<iframe width="100%" height="315" src="{yt_embed}" frameborder="0" allowfullscreen style="border-radius:12px;margin:16px 0;"></iframe>
"""
    course_block = ""
    if course_link:
        course_block = f'Include this course link: <a href="{course_link}" target="_blank" style="color:#06b6d4;font-weight:700;">📚 Official Certification Course →</a>'

    prompt = f"""
You are WhatNxt AI Career Mentor. Create a DETAILED, VISUAL, HTML-formatted career roadmap.

STUDENT: {name} | CURRENT YEAR: {standard} | TARGET ROLE: {role}

CAREER DATA FROM DATABASE:
{career_ctx}

INSTRUCTIONS:
1. Create a year-by-year roadmap from their CURRENT year ({standard}) to Year 4.
2. Each year should be a styled <div> box with a colored left border.
3. Include specific skills, projects, certifications, and monthly milestones.
4. Show a "Target Job" box at the end with: {job_title} at {company}
5. Include a "Skills to Master" section with colored skill badges.
6. {youtube_block}
7. {course_block}
8. Add a "Top Companies Hiring" section.
9. Add salary expectation info.
10. Use inline CSS for all styling. Use colors like #3b82f6 (blue), #06b6d4 (cyan), #22c55e (green).
11. Do NOT use markdown. Return raw HTML only. No ```html wrapper.
Make it extremely detailed and visually stunning.
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    roadmap_html = response.text.replace("```html","").replace("```","").strip()
    log_progress(name, 'roadmap_generated', role)
    return jsonify({"roadmap": roadmap_html})

# ═══════════════════════════════════════════════════════════════
# RESUME BUILDER (RAG-enhanced)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/build_resume', methods=['POST'])
def build_resume():
    data = request.json
    name = data.get('name', 'Student')
    role = data.get('role', 'Software Engineer')
    career_ctx = get_career_context(role)

    prompt = f"""
Create a PROFESSIONAL, ATS-optimized HTML Resume for:
Name: {name}
Academic Year: {data.get('standard')}
GPA: {data.get('gpa')} / 10
Target Role: {role}

CAREER DATA (use to populate relevant skills and keywords):
{career_ctx}

RESUME REQUIREMENTS:
- Sections: Header, Professional Summary, Education, Technical Skills, Projects (2 realistic ones), Certifications, Achievements
- Use a clean white background (#fff), dark text (#1a1a1a)
- Professional font (Arial or Georgia), A4-like layout
- Skills section: use styled inline badge spans with light blue background
- Projects: include 2 realistic placeholder projects relevant to {role} with tech stack and impact
- Use blue (#2563eb) for headers and accents
- Include a fake but realistic college name (Anna University or VIT Vellore)
- ATS keywords from: {career_ctx[:300]}
- Do NOT use markdown. Return raw HTML with inline CSS only. No ```html wrapper.
Make it print-ready and beautiful.
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    resume_html = response.text.replace("```html","").replace("```","").strip()
    log_progress(name, 'resume_built', role)
    return jsonify({"resume": resume_html})

# ═══════════════════════════════════════════════════════════════
# CAREER DISCOVERY — Path 2 (for clueless students)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/discover_career', methods=['POST'])
def discover_career():
    data = request.json
    answers = data.get('answers', [])   # list of chosen option labels
    marks = data.get('marks', {})       # {"Maths": "A", "CS": "A", "Physics": "B"}
    name = data.get('name', 'Student')
    standard = data.get('standard', '1st Year BTech')

    # Simple pattern matching for career suggestion
    answer_str = "".join(answers)
    suggested = "Software Engineer"  # default

    career_scores = {
        "Software Engineer": 0, "Data Scientist": 0, "Cybersecurity Analyst": 0,
        "Cloud Architect": 0, "AI Engineer": 0, "Product Manager": 0,
        "DevOps Engineer": 0, "Mobile App Developer": 0
    }

    # Score based on answers
    for ans in answers:
        ans_lower = ans.lower()
        if any(k in ans_lower for k in ["puzzle","math","code","building","automate"]):
            career_scores["Software Engineer"] += 2
        if any(k in ans_lower for k in ["data","pattern","analysis","statistics","excel"]):
            career_scores["Data Scientist"] += 2
        if any(k in ans_lower for k in ["security","protect","vulnerability","hacking"]):
            career_scores["Cybersecurity Analyst"] += 2
        if any(k in ans_lower for k in ["cloud","server","infrastructure","deployment"]):
            career_scores["Cloud Architect"] += 2
        if any(k in ans_lower for k in ["ai","intelligent","learning","experiment","research"]):
            career_scores["AI Engineer"] += 2
        if any(k in ans_lower for k in ["people","manage","strategy","coordinate","plan"]):
            career_scores["Product Manager"] += 2
        if any(k in ans_lower for k in ["app","mobile","build","user","daily"]):
            career_scores["Mobile App Developer"] += 2

    # Factor in marks
    if marks.get('Maths') in ['A', 'A+', 'S', '9', '10']:
        career_scores["Data Scientist"] += 1
        career_scores["AI Engineer"] += 1
    if marks.get('CS') in ['A', 'A+', 'S', '9', '10']:
        career_scores["Software Engineer"] += 2

    suggested = max(career_scores, key=career_scores.get)
    career_ctx = get_career_context(suggested)

    prompt = f"""
You are WhatNxt AI Career Counselor.

STUDENT: {name} | YEAR: {standard}
PSYCHOMETRIC ANSWERS: {answers}
MARKS: {marks}
AI CAREER SUGGESTION: {suggested}
CAREER DATA: {career_ctx}

Generate an HTML career discovery report with:
1. A styled "Career Match" banner showing: {suggested} with a match percentage (make it realistic, 75-92%)
2. Explanation of WHY this career suits this student based on their answers
3. "Your Strength Signals" section with 4-5 points
4. A simplified 4-year roadmap (Year 1 through Year 4 as colored boxes)
5. "First 3 Steps to Take This Week" section
6. Motivational closing message
Use inline CSS, blue/cyan color scheme, no markdown, raw HTML only.
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    result_html = response.text.replace("```html","").replace("```","").strip()

    # Save career path to DB
    try:
        conn = sqlite3.connect('whatnxt.db')
        conn.execute("UPDATE users SET career_path=? WHERE username=?", (suggested, name))
        conn.commit()
        conn.close()
    except: pass

    log_progress(name, 'career_discovered', suggested)
    return jsonify({"career": suggested, "report": result_html, "score": career_scores})

# ═══════════════════════════════════════════════════════════════
# QUIZ ENGINE
# ═══════════════════════════════════════════════════════════════
@app.route('/api/get_quiz', methods=['POST'])
def get_quiz():
    data = request.json
    domain = data.get('domain', 'Software Engineer')
    count = int(data.get('count', 5))

    if df_quiz.empty:
        return jsonify({"error": "Quiz data not loaded"}), 500

    domain_q = df_quiz[df_quiz['Domain'] == domain]
    if domain_q.empty:
        domain_q = df_quiz[df_quiz['Domain'] != 'Psychometric']

    sample = domain_q.sample(min(count, len(domain_q)))
    questions = []
    for _, row in sample.iterrows():
        questions.append({
            "question": row['Question'],
            "options": {
                "A": row['Option_A'], "B": row['Option_B'],
                "C": row['Option_C'], "D": row['Option_D']
            },
            "correct": row['Correct'],
            "explanation": row['Explanation']
        })
    return jsonify({"questions": questions, "domain": domain, "total": len(questions)})

@app.route('/api/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    answers = data.get('answers', {})   # {"0": "B", "1": "C", ...}
    correct_answers = data.get('correct_answers', {})
    username = data.get('username', '')
    domain = data.get('domain', '')

    score = sum(1 for i, ans in answers.items() if correct_answers.get(str(i)) == ans)
    total = len(correct_answers)
    percentage = round((score / total) * 100) if total > 0 else 0

    # Grade
    if percentage >= 80:
        grade = "Excellent 🏆"; next_step = "You're ready for advanced topics!"
    elif percentage >= 60:
        grade = "Good 👍"; next_step = "Review the topics you missed and try again."
    elif percentage >= 40:
        grade = "Average 📚"; next_step = "Spend more time on fundamentals."
    else:
        grade = "Needs Work 💪"; next_step = "Start from the basics — check the courses section."

    log_progress(username, 'quiz_completed', f"{domain}:{score}/{total}")
    return jsonify({"score": score, "total": total, "percentage": percentage, "grade": grade, "next_step": next_step})

@app.route('/api/get_psychometric_quiz', methods=['GET'])
def get_psychometric_quiz():
    if df_quiz.empty:
        return jsonify({"error": "No data"}), 500
    psycho_q = df_quiz[df_quiz['Domain'] == 'Psychometric']
    questions = []
    for _, row in psycho_q.iterrows():
        questions.append({
            "question": row['Question'],
            "options": {"A": row['Option_A'], "B": row['Option_B'], "C": row['Option_C'], "D": row['Option_D']},
            "explanation": row['Explanation']
        })
    return jsonify({"questions": questions})

# ═══════════════════════════════════════════════════════════════
# JOBS BOARD
# ═══════════════════════════════════════════════════════════════
@app.route('/api/get_jobs', methods=['POST'])
def get_jobs():
    data = request.json
    role = data.get('role', '')
    job_type = data.get('type', '')     # Full-time / Internship / All
    location = data.get('location', '')

    if df_jobs.empty:
        return jsonify({"jobs": [], "total": 0})

    filtered = df_jobs.copy()

    if role and role != 'All':
        mask = filtered['Role'].str.contains(role.split()[0], case=False, na=False)
        filtered = filtered[mask]

    if job_type and job_type != 'All':
        filtered = filtered[filtered['Type'].str.contains(job_type, case=False, na=False)]

    if location and location != 'All':
        filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]

    jobs_list = filtered.to_dict(orient='records')
    return jsonify({"jobs": jobs_list, "total": len(jobs_list)})

# ═══════════════════════════════════════════════════════════════
# COURSES
# ═══════════════════════════════════════════════════════════════
@app.route('/api/get_courses', methods=['POST'])
def get_courses():
    data = request.json
    domain = data.get('domain', 'Software Engineer')
    level = data.get('level', 'All')

    if df_courses.empty:
        return jsonify({"courses": []})

    filtered = df_courses[df_courses['Domain'] == domain]
    if level and level != 'All':
        filtered = filtered[filtered['Level'].str.contains(level, case=False, na=False)]

    return jsonify({"courses": filtered.to_dict(orient='records')})

# ═══════════════════════════════════════════════════════════════
# YEARLY PLAN
# ═══════════════════════════════════════════════════════════════
@app.route('/api/yearly_plan', methods=['POST'])
def yearly_plan():
    data = request.json
    role = data.get('role', 'Software Engineer')
    standard = data.get('standard', '1st Year BTech')
    career_ctx = get_career_context(role)

    current_year_num = 1
    for i, yr in enumerate(["1st","2nd","3rd","4th"], 1):
        if yr in standard:
            current_year_num = i
            break

    plans = {}
    for col_suffix, yr_num in [("Year_1_Plan",1),("Year_2_Plan",2),("Year_3_Plan",3),("Year_4_Plan",4)]:
        rows = df_career[df_career['Target_Role'] == role] if not df_career.empty else pd.DataFrame()
        if not rows.empty and col_suffix in rows.columns:
            plans[f"Year {yr_num}"] = rows.iloc[0][col_suffix]

    return jsonify({
        "current_year": current_year_num,
        "role": role,
        "plans": plans,
        "remaining_years": list(range(current_year_num, 5))
    })

# ═══════════════════════════════════════════════════════════════
# AI CHAT (with career context injection)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    name = data.get('name', 'Student')
    career = data.get('career', '')
    history = data.get('history', [])

    career_ctx = get_career_context(career) if career else ""
    jobs_ctx   = get_jobs_context(career) if career else ""

    history_str = ""
    for h in history[-6:]:  # last 6 turns
        history_str += f"{h.get('sender','')}: {h.get('text','')}\n"

    prompt = f"""You are NxtBot, WhatNxt's AI Career Mentor for BTech CSE students.
Student Name: {name}
Target Career: {career or 'Not set yet'}

CAREER DATA:
{career_ctx[:800] if career_ctx else 'No career set yet'}

AVAILABLE JOBS:
{jobs_ctx[:500] if jobs_ctx else ''}

CONVERSATION HISTORY:
{history_str}

Student's message: {message}

Reply helpfully and concisely (max 3-4 sentences). Be encouraging, specific, and practical.
If asked about jobs, courses, or roadmap, reference the data above.
Do not use markdown formatting — plain text only.
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    log_progress(name, 'chat', message[:50])
    return jsonify({"reply": response.text.strip()})

# ═══════════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════
@app.route('/api/get_progress', methods=['POST'])
def get_progress():
    username = request.json.get('username', '')
    try:
        conn = sqlite3.connect('whatnxt.db')
        c = conn.cursor()
        c.execute("SELECT action, detail, timestamp FROM progress WHERE username=? ORDER BY timestamp DESC LIMIT 20", (username,))
        rows = c.fetchall()
        conn.close()
        progress = [{"action": r[0], "detail": r[1], "time": r[2]} for r in rows]
        return jsonify({"progress": progress})
    except Exception as e:
        return jsonify({"progress": [], "error": str(e)})

# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "datasets": {
            "career_data": len(df_career),
            "jobs": len(df_jobs),
            "quiz": len(df_quiz),
            "courses": len(df_courses),
        }
    })

if __name__ == '__main__':
    print("🚀 WhatNxt API Server starting on http://127.0.0.1:5000")
    app.run(port=5000, debug=True, use_reloader=False)
