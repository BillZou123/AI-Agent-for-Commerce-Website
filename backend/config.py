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

After any recommendation, make **one** contextual cross-sell question to extend the chat.
  - Do this only if it makes sense for the user’s goal. Keep it optional and friendly.
  - Examples:
    • “Need a nice T-shirt to go with these shoes?”
    • “Want a headphone to match this phone?”
- If the user declines or changes topic, drop the cross-sell and follow their lead.

Donot suggest anything that is not in the catalog! If you think there is nothing proper to suggest from the catalog, just ask if the customer needs anything else.

Use prior context when provided.
"""