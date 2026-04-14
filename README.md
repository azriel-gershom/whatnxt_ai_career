# 🚀 WhatNxt — Enterprise AI Career Portal

An intelligent career guidance platform for Indian engineering students, powered by **Google Gemini AI**. Features AI-generated roadmaps, resume building, career discovery quizzes, job matching, YouTube integration, and an AI chatbot mentor.

---

## 📁 Project Structure

```
Whatnxt/
├── app.py                  # Flask API server (all backend logic)
├── index.html              # Single-page frontend (HTML/CSS/JS)
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
│
├── data/                   # 📊 Datasets
│   ├── career_data.csv         # Career paths, skills, salary, plans
│   ├── courses_data.csv        # Curated course recommendations
│   ├── quiz_data.csv           # Domain-specific quiz questions
│   ├── psychometric_map.csv    # Psychometric assessment mapping
│   ├── jobs_data.csv           # Job listings (curated)
│   ├── job_data.csv            # Large Kaggle job dataset (~64MB)
│   ├── companies.csv           # Company directory
│   ├── company_industries.csv  # Industry classification
│   ├── company_specialities.csv# Company specializations
│   ├── employee_counts.csv     # Company size data
│   ├── benefits.csv            # Job benefits data
│   ├── job_industries.csv      # Job-to-industry mapping
│   └── job_skills.csv          # Job-to-skill mapping
│
├── db/                     # 🗄️ SQLite Databases
│   ├── whatnxt.db              # Main app database (users, progress)
│   └── users.db                # Legacy user database
│
├── scripts/                # 🔧 Utility Scripts
│   ├── fix_db.py               # DB migration — add missing columns
│   └── update_db.py            # Profile update code reference
│
└── archive/                # 📦 Legacy Patches (already integrated)
    ├── frontend_additions.html # HTML/CSS/JS additions (merged)
    ├── js_fixes.js             # JS patches (merged)
    └── css_addition.txt        # CSS patches (merged)
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend
python app.py

# 3. Open the frontend
# Open index.html in your browser
# The API runs at http://127.0.0.1:5000
```

---

## 🔑 API Keys

Edit `app.py` and add your keys:
- **Gemini AI**: Get free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **YouTube Data API** (optional): [Google Cloud Console](https://console.cloud.google.com)

---

## 🛠 Features

| Feature | Description |
|---------|-------------|
| 🗺️ **AI Roadmap** | Year-by-year career plan with embedded YouTube tutorials |
| 📄 **Resume Forge** | ATS-optimized HTML resume generator |
| 🧭 **Career Discovery** | Quiz-based career matching with detailed reports |
| 🧠 **Skill Quiz** | Domain-specific knowledge assessment |
| 💼 **Job Board** | AI-curated job listings with cover letter generation |
| 📚 **Course Library** | Curated free & paid course recommendations |
| 👁️ **Certificate Scanner** | Vision AI skill extraction from certificate images |
| 💬 **AI Chatbot** | Personal career mentor powered by Gemini |

---

*Built with Flask, Google Gemini AI, and ❤️ for Indian engineering students.*
