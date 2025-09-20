import os
import uuid
import requests
import PyPDF2
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List
import edge_tts
import pygame
import threading

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI   # ✅ Use official SDK
from langchain.schema import Document
from langchain_chroma import Chroma

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("TeachingAssistant")

load_dotenv()

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LESSON_FILE = os.path.join(BASE_DIR, "last_lesson.txt")
AUDIO_FILES_DIR = os.path.join(BASE_DIR, "audio_files")  # ✅ inside server/
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)


class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1500
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_docs: int = 5
    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"


# ---------------- Document Processor ----------------
class DocumentProcessor:
    @staticmethod
    def load_from_url(url: str) -> str:
        logger.info(f"Loading document from URL: {url}")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        for s in soup(["script", "style"]):
            s.decompose()
        text = " ".join(soup.stripped_strings)
        logger.info(f"Extracted {len(text)} characters from URL")
        return text

    @staticmethod
    def load_from_pdf(file_path: str) -> str:
        logger.info(f"Loading PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text

    @staticmethod
    def load_from_txt(file_path: str) -> str:
        logger.info(f"Loading TXT file: {file_path}")
        text = Path(file_path).read_text(encoding="utf-8")
        logger.info(f"Extracted {len(text)} characters from TXT")
        return text


# ---------------- RAG Pipeline ----------------
class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.vectorstore = None

    def process_document(self, text: str, source: str = "doc"):
        logger.info("Splitting document into chunks...")
        chunks = self.splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        docs = [Document(page_content=c, metadata={"source": source, "id": i}) for i, c in enumerate(chunks)]
        self.vectorstore = Chroma.from_documents(docs, embedding=self.embeddings)
        logger.info("Document embedded into Chroma vector store")
        return docs

    def retrieve_chunks(self, query: str) -> List[str]:
        if not self.vectorstore:
            logger.warning("No vectorstore available. Did you process_document first?")
            return []
        logger.info(f"Retrieving top {self.config.top_k_docs} chunks for query: {query}")
        results = self.vectorstore.similarity_search(query, k=self.config.top_k_docs)
        return [r.page_content for r in results]


# ---------------- LLM ----------------
class LLMProvider:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def generate(self, prompt: str) -> str:
        logger.info(f"Sending prompt to LLM (model={self.config.model_name}, max_tokens={self.config.max_tokens})")
        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "system", "content": "You are a teaching assistant."},
                      {"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        text = resp.choices[0].message.content
        logger.info(f"LLM returned {len(text)} characters")
        return text


# ---------------- Audio ----------------
class AudioManager:
    _files = []

    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.Lock()

    async def text_to_speech(self, text: str) -> str:
        import edge_tts
        # Save audio in the folder provided by config.audio_dir
        fname = os.path.join(self.config.audio_dir, f"{uuid.uuid4().hex}.mp3")
        com = edge_tts.Communicate(
            text, self.config.tts_voice, rate=self.config.tts_rate, volume=self.config.tts_volume
        )
        await com.save(fname)
        AudioManager._files.append(fname)
        return fname



# ---------------- Teaching Assistant ----------------
class TeachingAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.rag = RAGPipeline(config)
        self.llm = LLMProvider(config)
        self.audio = AudioManager(config)
        self.lesson_file = Path(LESSON_FILE)

    def save_lesson(self, text: str):
        logger.info("Saving lesson text to file")
        self.lesson_file.write_text(text, encoding="utf-8")

    def load_lesson(self) -> str:
        logger.info("Loading last saved lesson")
        return self.lesson_file.read_text(encoding="utf-8") if self.lesson_file.exists() else ""

    async def ingest_and_teach(self, file_path: str):
        logger.info(f"Ingesting file: {file_path}")

        # Detect type & extract text
        if file_path.endswith(".pdf"):
            text = DocumentProcessor.load_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            text = DocumentProcessor.load_from_txt(file_path)
        elif file_path.startswith("http"):
            text = DocumentProcessor.load_from_url(file_path)
        else:
            logger.error("Unsupported file type")
            raise ValueError("Unsupported file type")

        # Process doc into vector DB
        logger.info("Processing document into RAG pipeline...")
        self.rag.process_document(text)

        # Generate lesson
        logger.info("Generating lesson with LLM...")
        lesson = self.llm.generate(f"Create a clear spoken lesson from this document:\n\n{text[:4000]}")
        self.save_lesson(lesson)

        # Generate MP3 lesson audio
        logger.info("Converting lesson to speech...")
        audio_path = await self.audio.text_to_speech(lesson)
        logger.info(f"Lesson ready. Audio: {audio_path}")
        return audio_path, lesson

    def cleanup(self):
        logger.info("Cleaning up generated audio files...")
        for f in AudioManager._files:
            try:
                os.remove(f)
                logger.info(f"Deleted {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
        AudioManager._files.clear()
