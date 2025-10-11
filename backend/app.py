from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from openai import OpenAI
from collections import deque
import os, pathlib, uuid
import base64

# ------------ Config ------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing in backend/.env")

client = OpenAI(api_key=api_key)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR))

SYSTEM_PROMPT = (
    "You are a friendly AI agent. Keep answers concise (1–3 sentences). "
    "Use prior context when provided."
)

# ------------ Session Memory ------------
HISTORY = {}                 # { session_id: deque([...]) }
MAX_TURNS = 50 


def get_history(session_id):
    if session_id not in HISTORY:
        HISTORY[session_id] = deque(maxlen=MAX_TURNS)
    return HISTORY[session_id]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ------------ Frontend ------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")

# Serve uploaded files so the browser (and model) can access them
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

# ------------ Image Upload ------------
# Expect multipart/form-data with field name "image"
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image'"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(f.filename)
    dest = UPLOAD_DIR / filename

    # Deduplicate filenames
    i = 1
    stem, suffix = dest.stem, dest.suffix
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{i}{suffix}"
        i += 1

    f.save(dest)
    public_url = f"/uploads/{dest.name}"   # relative URL browser can use
    return jsonify({"ok": True, "url": public_url})

# ------------ Chat (text or text+image) ------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg  = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    image_url = data.get("image_url")  # optional: e.g. "/uploads/xxx.jpg"

    if not user_msg and not image_url:
        return jsonify({"reply": "Please send a message or attach an image.",
                        "session_id": session_id})

    hist = get_history(session_id)

    # Build OpenAI messages
    # If there's an image, we send multi-part content to the model.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include past turns
    messages.extend(list(hist))

    if image_url:
        
        if image_url.startswith("/"):
            img_path = UPLOAD_DIR / pathlib.Path(image_url).name
        else:
            img_path = ROOT / pathlib.Path(image_url)
        encoded_image = encode_image(img_path)

        user_content = []
        if user_msg:
            user_content.append({"type": "text", "text": user_msg})
        user_content.append({"type": "image_url", 
                             "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_msg})

    # Call model
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
    )
    reply = resp.choices[0].message.content

    # Update memory
    # Store a compact marker for image so history stays readable
    if image_url:
        hist.append({"role": "user", "content": user_content})
    else:
        hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "session_id": session_id})

# ------------ Dev Server ------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)