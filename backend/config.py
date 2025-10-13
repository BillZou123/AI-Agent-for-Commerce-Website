import os, pathlib
from dotenv import load_dotenv
import json

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

# ----- Load catalog -----
with open(CATALOG_JSON, "r") as f:
    catalog_data = json.load(f)

catalog_text = json.dumps(catalog_data, indent=2)


# ----- System prompt for chat model -----
SYSTEM_PROMPT = f"""
You are a friendly AI agent for a commerce website. 

The user may chat with you in two ways:
1. General conversation.
2. Ask you to recommend or search for products.

You can only recommend products from the following catalog:
{catalog_text}

Keep the conversation engaging and useful so the user is happy to stay and explore more. 
You can make cross-sell questions by looking at catalog, and recommend something proper.
Use prior context when provided。
"""