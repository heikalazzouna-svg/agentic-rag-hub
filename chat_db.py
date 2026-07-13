import sqlite3
import json
from pathlib import Path

DB_PATH = Path("chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            chunks TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_message(role: str, content: str, sources: list = None, chunks: list = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sources_json = json.dumps(sources) if sources else "[]"
    chunks_json = json.dumps(chunks) if chunks else "[]"
    
    cursor.execute(
        "INSERT INTO messages (role, content, sources, chunks) VALUES (?, ?, ?, ?)",
        (role, content, sources_json, chunks_json)
    )
    conn.commit()
    conn.close()

def get_messages() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, sources, chunks FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        role, content, sources_json, chunks_json = row
        messages.append({
            "role": role,
            "content": content,
            "sources": json.loads(sources_json) if sources_json else [],
            "chunks": json.loads(chunks_json) if chunks_json else []
        })
    return messages

def clear_chat():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

# Initialize the database on import
init_db()
