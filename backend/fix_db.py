import sqlite3

conn = sqlite3.connect('whatnxt.db')
try:
    conn.execute("ALTER TABLE users ADD COLUMN career_path TEXT")
    conn.execute("ALTER TABLE users ADD COLUMN quiz_scores TEXT")
    conn.commit()
    print("✅ Columns added successfully!")
except Exception as e:
    print(f"Note: {e}")
conn.close()