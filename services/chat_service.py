import os, io, logging
from typing import Dict, List
from openai import OpenAI

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# OpenAI client
client = None
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")

# In-memory conversations
conversations: Dict[str, List[Dict[str, str]]] = {}

# --- Helpers ---
def get_last_assistant_text(uid: str) -> str:
    msgs = conversations.get(uid, [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return (m.get("content") or "").replace("<END_CONVO>", "").strip()
    return ""

def start_conversation(uid: str, name: str, dept: str, college: str) -> bytes:
    """Start new conversation and return audio bytes."""
    if not client:
        raise RuntimeError("OpenAI client not initialized.")

    messages = [
        {"role": "system", "content": (
            "You are 'Nova', a friendly and enthusiastic FEMALE voice assistant at AWS Day. "
            "Your goal is to warmly welcome participants, discuss cloud technology, career opportunities, and AWS services. "
            "Keep responses engaging, natural, supportive, and concise. "
            "When the conversation should end, finish with a goodbye and append <END_CONVO>."
        )},
        {"role": "user", "content": f"Hi! My name is {name}, and I'm from the {dept} department at {college}."}
    ]
    conversations[uid] = messages

    chat_resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = chat_resp.choices[0].message.content
    conversations[uid].append({"role": "assistant", "content": reply})

    # Convert to speech
    is_end = "<END_CONVO>" in reply
    speak_text = reply.replace("<END_CONVO>", "").strip()
    audio_resp = client.audio.speech.create(model="tts-1", voice="nova", input=speak_text)
    return audio_resp.read(), is_end

def continue_conversation(uid: str, user_msg: str) -> bytes:
    """Continue conversation with a user message."""
    if not client:
        raise RuntimeError("OpenAI client not initialized.")
    if uid not in conversations:
        raise ValueError("Conversation not found.")

    conversations[uid].append({"role": "user", "content": user_msg})
    chat_resp = client.chat.completions.create(model="gpt-4o-mini", messages=conversations[uid])
    reply = chat_resp.choices[0].message.content
    conversations[uid].append({"role": "assistant", "content": reply})

    is_end = "<END_CONVO>" in reply
    speak_text = reply.replace("<END_CONVO>", "").strip()
    audio_resp = client.audio.speech.create(model="tts-1", voice="nova", input=speak_text)
    return audio_resp.read(), is_end
