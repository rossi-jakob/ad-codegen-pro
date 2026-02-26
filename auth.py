"""
Simple Authentication System for Streamlit
Uses SQLite + Password Hashing
"""

import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path("users.db")


class AuthManager:

    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str) -> bool:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, self._hash_password(password))
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def login(self, username: str, password: str) -> bool:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False

        return row[0] == self._hash_password(password)