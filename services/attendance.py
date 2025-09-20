from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import sqlite3
from datetime import date
from contextlib import asynccontextmanager
from gtts import gTTS
import os

DB_NAME = "attendance.db"
router = APIRouter()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT,
        name TEXT,
        reg_no TEXT,
        dept TEXT,
        extra_info TEXT,
        date TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(roll_no, date)
    )
    """)
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app):
    print("🚀 Initializing database...")
    init_db()
    yield
    print("🛑 Shutting down...")

class QRData(BaseModel):
    qr_data: str

def generate_voice(text: str, filename: str = "attendance.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

@router.post("/mark-attendance")
async def mark_attendance(data: QRData):
    try:
        roll_no, name, reg_no, dept, extra_info = data.qr_data.split(",", 4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid QR format")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO attendance (roll_no, name, reg_no, dept, extra_info, date)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (roll_no, name, reg_no, dept, extra_info, date.today()))
        conn.commit()
        status = "success"
    except sqlite3.IntegrityError:
        status = "marked"

    conn.close()

    # Generate MP3 voice
    if status == "success":
        msg = f"{name}, your attendance has been marked."
    else:
        msg = f"{name}, you have already marked attendance today."
    
    mp3_file = generate_voice(msg, filename=f"{roll_no}_attendance.mp3")

    # Return both JSON and MP3 path (frontend can fetch / play)
    return {
        "status": status,
        "name": name,
        "voice_file": mp3_file  # frontend can fetch this file
    }

# Optional: Serve generated MP3s
@router.get("/voice/{filename}")
async def get_voice(filename: str):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="File not found")
