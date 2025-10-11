import os, pathlib
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing in backend/.env")

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CATALOG_JSON = ROOT / "backend" / "data" / "catalog.json"

SYSTEM_PROMPT = (
    "You are a friendly AI agent. Keep answers concise (1–3 sentences). "
    "Use prior context when provided."
)