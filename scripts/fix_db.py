import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / 'db' / 'whatnxt.db'
conn = sqlite3.connect(str(DB_PATH))
try:
    conn.execute("ALTER TABLE users ADD COLUMN career_path TEXT")
    conn.execute("ALTER TABLE users ADD COLUMN quiz_scores TEXT")
    conn.commit()
    print("✅ Columns added successfully!")
except Exception as e:
    print(f"Note: {e}")
conn.close()