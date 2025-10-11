import os, pathlib
from dotenv import load_dotenv

# ----- Environment setup -----
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing in backend/.env")

# ----- Project paths -----
ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CATALOG_JSON = ROOT / "backend" / "data" / "catalog.json"

# ----- System prompt for chat model -----
SYSTEM_PROMPT = (
    "You are a friendly AI agent. Keep answers concise. "
    "Use prior context when provided."
)