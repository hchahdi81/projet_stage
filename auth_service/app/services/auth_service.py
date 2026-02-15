from utils.hashing import hash_password, verify_password
from db.connection import get_connection

def create_user(user):
    hashed_password = hash_password(user.password)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, email, password_hash)
        VALUES (%s, %s, %s)
    """, (user.name, user.email, hashed_password))
    conn.commit()
    cursor.close()
    conn.close()

def authenticate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, password_hash FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and verify_password(password, user[2]):
        return {"id": user[0], "name": user[1], "email": email}
    return None
