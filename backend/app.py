from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
import uuid, pathlib, json, re

from .config import FRONTEND_DIR, UPLOAD_DIR, SYSTEM_PROMPT, api_key
from .utils import allowed_file, encode_image
from .memory import get_history

client = OpenAI(api_key=api_key)
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# --- Serve frontend ---
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

# --- Handle image uploads ---
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
    i, stem, suffix = 1, dest.stem, dest.suffix
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{i}{suffix}"
        i += 1
    f.save(dest)
    return jsonify({"ok": True, "url": f"/uploads/{dest.name}"})


# --- Extract product items from model reply ---
_JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def extract_items_from_reply(text):
    m = _JSON_BLOCK.search(text or "")
    if not m:
        return []
    try:
        obj = json.loads(m.group(1))
        items = obj.get("items") or []
        return [
            {
                "id": it.get("id"),
                "name": it.get("name"),
                "price": it.get("price"),
                "category": it.get("category"),
                "image": it.get("image"),
            }
            for it in items
            if isinstance(it, dict) and "id" in it and "name" in it
        ]
    except Exception:
        return []


# --- Chat endpoint  ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    image_url = data.get("image_url")

    if not user_msg and not image_url:
        return jsonify({"reply": "Please send a message or attach an image.",
                        "session_id": session_id})

    hist = get_history(session_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    schema_hint = (
        "If your response contain any products from the catalog, after your reply append a fenced JSON block:\n"
        "```json\n{ \"items\": [ {\"id\":\"...\",\"name\":\"...\",\"price\":0,\"category\":\"...\",\"image\":image_path} ] }\n```\n"
        "Only include JSON if your response actually contain product items from the catalog."

        "You can only include file paths inside the JSON block, for example when the user is asking for an image of a product, itstead of showing the img path in our text reply, just append the JSON block of that product in your reply, so that my frontend will render the image."

    )
    messages.append({"role": "system", "content": schema_hint})
    messages.extend(list(hist))

    if image_url:
        if image_url.startswith("/"):
            img_path = UPLOAD_DIR / pathlib.Path(image_url).name
        else:
            img_path = pathlib.Path(image_url)
        encoded = encode_image(img_path)
        user_content = []
        if user_msg:
            user_content.append({"type":"text","text": user_msg})
        user_content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encoded}"}})
        messages.append({"role":"user","content": user_content})
    else:
        messages.append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
    )
    reply_text = resp.choices[0].message.content or ""
    items = extract_items_from_reply(reply_text)

    if image_url:
        hist.append({"role": "user", "content": user_content})
    else:
        hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply_text})

    print(reply_text)
    return jsonify({"reply": reply_text, "session_id": session_id, "items": items})


# --- Run server ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)