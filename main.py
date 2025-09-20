import os
import io
import json
import logging
import tempfile
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- External libs ----------------
from openai import OpenAI

# ---------------- Internal services ----------------
from services.attendance import router, lifespan
from services.teaching import TeachingAssistant, Config
#from services.wakeword import detect_wakeword_whisper, chat_with_openai_messages, speak
 
# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILES_DIR = os.path.join(BASE_DIR, "services", "audio_files")
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

config = Config()
config.audio_dir = AUDIO_FILES_DIR
assistant = TeachingAssistant(config)

# ---------------- FastAPI App ----------------
app = FastAPI(lifespan=lifespan)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ Restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-UID", "X-Conversation-End", "X-Response-Length"]
)

# ---------------- Static Files ----------------
app.mount("/audio_files", StaticFiles(directory=AUDIO_FILES_DIR), name="audio_files")

# ---------------- Attendance Routes ----------------
app.include_router(router)

# ======================================================
# 📌 TEACHING ENDPOINTS
# ======================================================
@app.post("/teach_document/")
async def teach_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(BASE_DIR, f"uploaded_{uuid4().hex}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        audio_path, lesson_text = await assistant.ingest_and_teach(file_path)

        audio_filename = os.path.basename(audio_path)
        audio_url = f"http://127.0.0.1:8000/audio_files/{audio_filename}"

        return JSONResponse({
            "status": "success",
            "lesson_preview": lesson_text[:500],
            "audio_url": audio_url
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/cleanup_audio_files/")
def cleanup_audio_files():
    deleted = []
    for f in os.listdir(AUDIO_FILES_DIR):
        if f.endswith((".mp3", ".wav")):
            os.remove(os.path.join(AUDIO_FILES_DIR, f))
            deleted.append(f)
    return {"deleted_files": deleted, "status": "Audio files cleaned!"}

# ======================================================
# 📌 VOICE ASSISTANT (Nephele) ENDPOINTS
# ======================================================

# ---- OpenAI Client Setup ----
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("❌ OPENAI_API_KEY not found in environment variables!")
        client = None
    else:
        client = OpenAI(api_key=api_key)
        logging.info("✅ OpenAI client initialized")
except Exception as e:
    logging.error(f"❌ Failed to initialize OpenAI client: {e}")
    client = None

MODEL = "gpt-4o-mini"

# ---- Pydantic Models ----
class QRData(BaseModel):
    qr_data: str

class ChatData(BaseModel):
    uid: str
    message: str

# ---- Globals ----
voice_conversations = {}

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Nephele 3.0 API</title></head>
    <body>
        <h1>Nephele 3.0 Central API</h1>
        <p>✅ Backend is running!</p>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/voice/active_sessions">Active Voice Sessions</a></li>
        </ul>
    </body>
    </html>
    """)


@app.post("/voice/process_qr/")
async def voice_process_qr(data: QRData):
    """Initialize voice conversation from QR code data"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

    qr_parts = data.qr_data.split(",")
    if len(qr_parts) < 5:
        raise HTTPException(status_code=400, detail="Invalid QR code format")

    uid, name, reg_no, dept, college = qr_parts[:5]
    uid = uid.strip()
    name = name.strip()
    dept = dept.strip()
    college = college.strip()

    messages = [
        {"role": "system", "content": (
            "You are 'Nova', a friendly FEMALE voice assistant at AWS Day. "
            "Engage users with cloud career opportunities, AWS services, "
            "and end with <END_CONVO> when done."
        )},
        {"role": "user", "content": f"Hi! I'm {name} from {dept} at {college}. ID: {reg_no}."}
    ]

    voice_conversations[uid] = messages

    chat_resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=120,
        temperature=0.7
    )
    reply = chat_resp.choices[0].message.content
    voice_conversations[uid].append({"role": "assistant", "content": reply})

    is_end = "<END_CONVO>" in reply
    speak_text = reply.replace("<END_CONVO>", "").strip()

    audio_resp = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=speak_text,
        speed=1.0
    )

    response = StreamingResponse(io.BytesIO(audio_resp.read()), media_type="audio/mpeg")
    response.headers["X-UID"] = str(uid)
    response.headers["X-Conversation-End"] = "true" if is_end else "false"
    response.headers["X-Response-Length"] = str(len(speak_text))
    return response


@app.post("/voice/chat/")
async def voice_chat(data: ChatData):
    """Continue voice conversation"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")
    if data.uid not in voice_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found. Please scan QR again.")

    voice_conversations[data.uid].append({"role": "user", "content": data.message})

    chat_resp = client.chat.completions.create(
        model=MODEL,
        messages=voice_conversations[data.uid],
        max_tokens=120,
        temperature=0.7
    )
    reply = chat_resp.choices[0].message.content
    voice_conversations[data.uid].append({"role": "assistant", "content": reply})

    is_end = "<END_CONVO>" in reply
    speak_text = reply.replace("<END_CONVO>", "").strip()

    audio_resp = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=speak_text,
        speed=1.0
    )

    response = StreamingResponse(io.BytesIO(audio_resp.read()), media_type="audio/mpeg")
    response.headers["X-UID"] = str(data.uid)
    response.headers["X-Conversation-End"] = "true" if is_end else "false"
    return response


@app.post("/voice/transcribe/")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return {"text": transcription.strip()}
    finally:
        os.unlink(temp_file_path)


@app.get("/voice/latest_text")
async def voice_latest_text(uid: str):
    """Get latest assistant text"""
    if uid not in voice_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    for message in reversed(voice_conversations[uid]):
        if message.get("role") == "assistant":
            return {"text": message.get("content", "").replace("<END_CONVO>", "").strip()}
    return {"text": "No assistant response found"}


@app.get("/voice/active_sessions")
async def get_active_voice_sessions():
    return {"active_sessions": list(voice_conversations.keys()), "total_sessions": len(voice_conversations)}


@app.delete("/voice/conversation/{uid}")
async def end_voice_conversation(uid: str):
    if uid in voice_conversations:
        del voice_conversations[uid]
        return {"message": "Conversation ended"}
    raise HTTPException(status_code=404, detail="Conversation not found")


""" 
# ======================================================
# INTERACTION AND WAKE WORRD
# ======================================================

conversations = {}


@app.post("/wakeword")
async def wakeword(audio: UploadFile = File(...)):
    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await audio.read())
    tmp.close()
    detected = detect_wakeword_whisper(tmp.name)
    os.unlink(tmp.name)
    return {"wake_detected": detected}

@app.post("/voice/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await audio.read())
    tmp.close()
    from wakeword import whisper_model
    result = whisper_model.transcribe(tmp.name)
    os.unlink(tmp.name)
    return {"text": result.get("text", "")}

@app.post("/voice/chat")
async def voice_chat(payload: dict):
    uid = payload.get("uid")
    msg = payload.get("message")
    if uid not in conversations:
        conversations[uid] = [{"role": "system", "content": "You are Nephele, a friendly AI assistant."}]
    conversations[uid].append({"role": "user", "content": msg})
    reply = chat_with_openai_messages(conversations[uid])
    conversations[uid].append({"role": "assistant", "content": reply})
    # Send TTS audio back to frontend
    tmp_file = speak(reply, use_openai=True)
    from fastapi.responses import FileResponse
    return FileResponse(tmp_file, media_type="audio/mpeg")

@app.get("/voice/latest_text")
async def latest_text(uid: str):
    if uid in conversations and len(conversations[uid]) > 0:
        return {"text": conversations[uid][-1]["content"]}
    return {"text": ""}

 """

# ======================================================
# SERVER ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting Central Backend (Attendance + Teaching + Voice)...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
