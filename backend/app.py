from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
import uuid, pathlib

# Local modules
from .config import FRONTEND_DIR, UPLOAD_DIR, SYSTEM_PROMPT, api_key
from .utils import allowed_file, encode_image
from .memory import get_history
from .catalog import search_catalog_semantic, format_products_text
from .intent import classify_intent_llm, image_to_query, map_term_to_tags_with_llm



client = OpenAI(api_key=api_key)

# Serve /catalog_images/* directly from the frontend folder
# because static_folder is FRONTEND_DIR and static_url_path=""
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# ---------- Frontend ----------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")

# Make uploaded files accessible (model + browser)
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

# ---------- Image Upload ----------
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
    public_url = f"/uploads/{dest.name}"
    return jsonify({"ok": True, "url": public_url})

# ---------- Chat (text or text+image) ----------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg  = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    image_url = data.get("image_url")  # optional

    if not user_msg and not image_url:
        return jsonify({"reply": "Please send a message or attach an image.",
                        "session_id": session_id})

    hist = get_history(session_id)

    # Decide intent with LLM
    intent = classify_intent_llm(user_msg, bool(image_url), image_url=image_url)

    # --- Branch: recommendation via IMAGE (image-only or image+text)
    if intent == "RECOMMEND_IMAGE":
        q_img = image_to_query(image_url) if image_url else ""
        query = f"{user_msg} {q_img}".strip() if user_msg else (q_img or "product")
        items = search_catalog_semantic(user_msg or q_img, tag_mapper=map_term_to_tags_with_llm, topk=5)
        reply = f"(query: {query})\n" + format_products_text(items)

        # Store user turn (image + optional text)
        if image_url:
            # embed the image in the history like your original logic
            if image_url.startswith("/"):
                img_path = UPLOAD_DIR / pathlib.Path(image_url).name
            else:
                img_path = pathlib.Path(image_url)
            encoded = encode_image(img_path)
            parts = []
            if user_msg:
                parts.append({"type":"text", "text": user_msg})
            parts.append({"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{encoded}"}})
            hist.append({"role":"user","content": parts})
        else:
            hist.append({"role":"user","content": user_msg or "(image)"})

        hist.append({"role":"assistant","content": reply})
        return jsonify({"reply": reply, "session_id": session_id})

    # --- Branch: recommendation via TEXT
    if intent == "RECOMMEND_TEXT":
        items = search_catalog_semantic(user_msg, tag_mapper=map_term_to_tags_with_llm, topk=5)
        reply = format_products_text(items)
        hist.append({"role":"user","content": user_msg})
        hist.append({"role":"assistant","content": reply})
        return jsonify({"reply": reply, "session_id": session_id})

    # --- Branch: general chat (text or image)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        messages.append({"role":"user","content": user_msg})

    # Call model for normal chat
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
    )
    reply = resp.choices[0].message.content

    # Update memory
    if image_url:
        hist.append({"role":"user","content": user_content})
    else:
        hist.append({"role":"user","content": user_msg})
    hist.append({"role":"assistant","content": reply})

    return jsonify({"reply": reply, "session_id": session_id})

# Optional: return catalog for debugging
from .catalog import CATALOG  # late import to avoid circularity
@app.route("/catalog", methods=["GET"])
def get_catalog():
    return jsonify(CATALOG)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)