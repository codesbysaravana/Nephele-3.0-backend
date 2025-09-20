import io, logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.chat_service import chat_service

router = APIRouter(tags=["Chat"])

class QRData(BaseModel):
    qr_data: str

class ChatData(BaseModel):
    uid: str
    message: str

@router.post("/process_qr/")
async def process_qr(data: QRData):
    try:
        parts = data.qr_data.split(",")
        if len(parts) < 5:
            raise HTTPException(status_code=400, detail="QR data must contain 5 comma-separated values")
        uid, name, _, dept, college = parts[:5]
        audio_bytes, is_end = chat_service.start_conversation(uid, name, dept, college)
        headers = {"X-Conversation-End": "true" if is_end else "false", "X-UID": uid}
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg", headers=headers)
    except Exception as e:
        logging.error(f"Error processing QR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/")
async def chat(data: ChatData):
    try:
        audio_bytes, is_end = chat_service.continue_conversation(data.uid, data.message)
        headers = {"X-Conversation-End": "true" if is_end else "false", "X-UID": data.uid}
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg", headers=headers)
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest_text")
async def latest_text(uid: str):
    return {"text": chat_service.get_last_assistant_text(uid)}
