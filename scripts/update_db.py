#!/usr/bin/env python3
"""
WhatNxt — Database Migration / Repair Script
Run this standalone to ensure the users table has all required columns.
Usage: python3 scripts/update_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / 'db' / 'whatnxt.db'

REQUIRED_COLUMNS = {
    'username':     'TEXT UNIQUE',
    'password':     'TEXT',
    'standard':     'TEXT',
    'gpa':          'REAL',
    'goals':        'TEXT',
    'career_path':  'TEXT',
    'quiz_scores':  'TEXT',
    'gender':       'TEXT',
    'dob':          'TEXT',
    'college':      'TEXT',
    'department':   'TEXT',
    'maths_grade':  'TEXT',
    'cs_grade':     'TEXT',
    'physics_grade':'TEXT',
    'english_grade':'TEXT',
    'skill_level':  'TEXT',
    'path_choice':  'TEXT',
    'created_at':   'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
}

def migrate():
    print(f"📂 Database: {DB_PATH}")
    if not DB_PATH.exists():
        print("⚠️  Database file not found. Run app.py first to create it.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get existing columns
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(users)")}
    print(f"✅ Existing columns: {', '.join(sorted(existing))}")

    # Add missing columns
    added = []
    for col, dtype in REQUIRED_COLUMNS.items():
        if col not in existing:
            # Strip constraints like UNIQUE / DEFAULT for ALTER TABLE
            safe_type = dtype.split()[0]  # Just TEXT, REAL, TIMESTAMP
            try:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {safe_type}")
                added.append(col)
                print(f"  ➕ Added column: {col} ({safe_type})")
            except Exception as e:
                print(f"  ⚠️  Could not add {col}: {e}")

    if added:
        conn.commit()
        print(f"\n✅ Migration complete — added {len(added)} column(s): {', '.join(added)}")
    else:
        print("\n✅ All columns already exist — nothing to do.")

    # Also ensure the progress table exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS progress (
        username TEXT, action TEXT, detail TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

    # Show row count
    count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"👥 Total users in database: {count}")

    conn.close()
    print("🏁 Done.")

if __name__ == '__main__':
    migrate()