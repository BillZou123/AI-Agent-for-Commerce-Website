from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
from collections import deque
import os, pathlib, uuid

# --- Load env ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing in backend/.env")

client = OpenAI(api_key=api_key)

# --- Flask app setup ---
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR))

SYSTEM_PROMPT = (
    "You are a friendly AI agent. Keep answers concise. "
    "Use prior context when provided."
)

# --- In-memory chat history: { session_id: deque([...]) } ---
# Each entry in deque is a dict: {"role": "user"|"assistant", "content": "text"}
HISTORY = {}
MAX_TURNS = 50  # total messages to retain (user+assistant pairs)

def get_history(session_id: str) -> deque:
    if session_id not in HISTORY:
        HISTORY[session_id] = deque(maxlen=MAX_TURNS)
    return HISTORY[session_id]

# --- Serve index.html ---
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# --- Chat endpoint with memory ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())  # fallback if none provided

    if not user_msg:
        return jsonify({"reply": "Please type a message.", "session_id": session_id})

    # Pull recent turns
    hist = get_history(session_id)

    # Build message list: system prompt + prior turns + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(hist))  # hist already in [{"role":..., "content":...}, ...] format
    messages.append({"role": "user", "content": user_msg})

    # Call model
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
    )
    reply = resp.choices[0].message.content

    # Update history
    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "session_id": session_id})

# --- Run server ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)