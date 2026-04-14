# ═══════════════════════════════════════════════════════════════
# UPDATE PROFILE (FIXED)
# ═══════════════════════════════════════════════════════════════
@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    data = request.json
    username = data.get('username')
    try:
        conn = sqlite3.connect('whatnxt.db')
        # Update ALL the new fields, not just the old ones
        conn.execute('''UPDATE users SET 
                        standard=?, gpa=?, goals=?, gender=?, dob=?, 
                        college=?, department=?, maths_grade=?, cs_grade=?, 
                        physics_grade=?, english_grade=?, skill_level=?, path_choice=? 
                        WHERE username=?''',
                     (data.get('standard',''), float(data.get('gpa', 0)), data.get('goals',''),
                      data.get('gender',''), data.get('dob',''), data.get('college',''),
                      data.get('department',''), data.get('maths_grade',''), data.get('cs_grade',''),
                      data.get('physics_grade',''), data.get('english_grade',''), data.get('skill_level',''),
                      data.get('path_choice',''), username))
        conn.commit()
        conn.close()
        log_progress(username, 'profile_update')
        return jsonify({"status":"success"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500