from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from openai import OpenAI
from collections import deque
import os, pathlib, uuid, re
import base64, json
import nltk
import string
from nltk.tokenize import word_tokenize

# ------------ Config ------------
nltk.download('punkt_tab')
nltk.download('punkt')
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ OPENAI_API_KEY missing in backend/.env")

client = OpenAI(api_key=api_key)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CATALOG_JSON = ROOT / "backend" / "data" / "catalog.json"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="" )

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

# ---------------- Utils ----------------
ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def _toks(s):
    if not s:
        return []
    tokens = word_tokenize(s.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    return tokens

LLM_TAG_CACHE = {}  # term -> set(tags)

def map_term_to_tags_with_llm(term):
    """
    Map one user term to 1–5 tags chosen ONLY from TAG_VOCAB.
    Results are cached to save latency/cost.
    """
    term = _norm(term)
    if not term:
        return set()
    if term in LLM_TAG_CACHE:
        return LLM_TAG_CACHE[term]

    sys = (
        "You map user terms to tags from a fixed vocabulary for product search. "
        "Return ONLY tags that appear in the provided vocabulary. "
        "Output a comma-separated list of 1-5 tags; return 'none' if nothing fits."
    )
    # If your tag list grows very large, you can slice it to keep the prompt short.
    vocab_str = ", ".join(TAG_VOCAB[:400])

    user = (
        "term: {t}\n"
        "allowed_tags: {v}\n"
        "respond with tags only."
    ).format(t=term, v=vocab_str)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:
        print("tag map llm error:", e)
        text = "none"

    # Keep only valid tags that appear in TAG_VOCAB
    candidates = [_norm(x) for x in re.split(r"[,\n]+", text)]
    tags = set([c for c in candidates if c in TAG_VOCAB])

    LLM_TAG_CACHE[term] = tags
    return tags
# ---------------- Catalog loading & indexing ----------------
# Load catalog JSON and build lightweight keyword index
with open(CATALOG_JSON, "r") as f:
    RAW_CATALOG = json.load(f)

CATALOG = []
for item in RAW_CATALOG:
    # build keywords from name/category/brand/tags 
    kw = set(_toks(item.get("name"))) \
       | set(_toks(item.get("category"))) \
       | set(_toks(item.get("brand"))) \
       | set().union(*[set(_toks(t)) for t in item.get("tags", [])])
    item["keywords"] = sorted(list(kw))
    CATALOG.append(item)

# Build a tag vocabulary for product recommendation
def _norm(s):
    return (s or "").strip().lower()

TAG_VOCAB = sorted(set(
    _norm(tag)
    for it in CATALOG
    for tag in it.get("tags", []) if tag is not None
))

def search_catalog(query, topk=5):
    """
    Simple token-overlap ranking over the prebuilt keywords.
    """
    q = set(_toks(query))
    ranked = []
    for it in CATALOG:
        kw = set(it.get("keywords", []))
        score = len(q & kw)
        # small boosts for exact brand/category mentions
        brand = (it.get("brand") or "").lower()
        cat   = (it.get("category") or "").lower()
        if brand in q: score += 2
        if cat in q:   score += 1
        if score > 0:
            ranked.append((score, it))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in ranked[:int(topk)]]

def search_catalog_semantic(query, topk=5):
    """
    LLM-assisted tag search:
    1) tokenize query
    2) map each term to catalog tags via LLM
    3) score items by tag overlap (+ small brand/category boosts)
    4) fallback to old token-overlap search if nothing found
    """
    terms = _toks(query)
    tag_candidates = set()
    for t in terms:
        tag_candidates |= map_term_to_tags_with_llm(t)

    ranked = []
    for it in CATALOG:
        item_tags = set(_norm(t) for t in it.get("tags", []))
        score = len(tag_candidates & item_tags)

        brand = _norm(it.get("brand"))
        cat   = _norm(it.get("category"))
        if brand and brand in terms: score += 2
        if cat   and cat   in terms: score += 1

        if score > 0:
            ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    hits = [it for _, it in ranked[:int(topk)]]

    # Fallback to the old method if nothing matched
    if not hits:
        try:
            hits = search_catalog(query, topk=topk)
        except Exception:
            pass

    return hits

def format_products_text(items):
    """
    Make a numbered, readable block. Example:

    Here are some picks from my catalog:
    1. **Apple MacBook Pro 14 (M2)** — $1999 · Apple (image: /catalog_images/mbp14.jpg)

    2. **Dell XPS 13** — $1299 · Dell (image: /catalog_images/xps13.jpg)
    """
    if not items:
        return "Sorry, I couldn’t find a match in my catalog."

    lines = ["Here are some picks from my catalog:"]
    for idx, it in enumerate(items, 1):
        name  = it.get("name", "Unknown")
        price = it.get("price")
        brand = (it.get("brand") or "").title()
        img   = it.get("image") or ""

        if price is not None:
            line = f"{idx}. **{name}** — ${price} · {brand} (image: {img})"
        else:
            line = f"{idx}. **{name}** · {brand} (image: {img})"

        lines.append(line)
        lines.append("")  # blank line between items

    return "\n".join(lines).rstrip()

# ---------------- LLM-Based Intent Detection ----------------
def classify_intent_llm(user_text, has_image, image_url=None):
    """
    Vision-aware intent classifier.
    Returns exactly one of: RECOMMEND_TEXT, RECOMMEND_IMAGE, CHAT_TEXT, CHAT_IMAGE
    """
    sys_msg = (
        "You are an intent classifier for a shopping assistant.\n"
        "Choose exactly one label:\n"
        " - RECOMMEND_TEXT: user asks to recommend/find/search products using TEXT only.\n"
        " - RECOMMEND_IMAGE: user wants recommendation/search based on the attached IMAGE (image-only or image+text).\n"
        " - CHAT_TEXT: general chat (no shopping intent), text only.\n"
        " - CHAT_IMAGE: general chat with an image (describe/ask about image, not searching catalog).\n"
        "\n"
        "Important rules:\n"
        "- Classify as RECOMMEND_IMAGE only if the image clearly shows a consumer product (e.g., shoes, clothes, bag, phone, laptop, headphones, camera, TV, vacuum, watch, etc.).\n"
        "- If the image is a chart/graph (e.g., stock chart), document/screenshot, scenery, face/selfie, or anything not obviously a shopping product -> classify as CHAT_IMAGE.\n"
        "- Return ONLY the label token."
    )

    # Always send a short structured text summary
    text_summary = f"has_image={str(bool(has_image)).lower()}\ntext={user_text or '(empty)'}"

    # Build multi-part user content; attach the actual image if provided
    content = [{"type": "text", "text": text_summary}]
    if image_url:
        # resolve path -> data URI
        if image_url.startswith("/"):
            img_path = UPLOAD_DIR / pathlib.Path(image_url).name
        else:
            img_path = ROOT / pathlib.Path(image_url)
        b64 = encode_image(img_path)
        data_uri = f"data:image/jpeg;base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": content},
            ],
            temperature=0.0,
        )
        label = (resp.choices[0].message.content or "").strip().upper()
        valid = {"RECOMMEND_TEXT","RECOMMEND_IMAGE","CHAT_TEXT","CHAT_IMAGE"}
        if label not in valid:
            # Safe fallback
            label = "CHAT_IMAGE" if has_image else "CHAT_TEXT"
        
        print("***************")
        print(label)
        return label
    except Exception as e:
        print("Intent classification error:", e)
        return "CHAT_IMAGE" if has_image else "CHAT_TEXT"

def image_to_query(image_url):
    """
    Convert an uploaded image to compact keywords via the vision model.
    Returns a short string like 'nike sneakers running' for catalog search.
    """
    if not image_url:
        return ""
    # resolve file path then to data: URI
    if image_url.startswith("/"):
        img_path = UPLOAD_DIR / pathlib.Path(image_url).name
    else:
        img_path = ROOT / pathlib.Path(image_url)
    b64 = encode_image(img_path)
    data_uri = "data:image/jpeg;base64,{0}".format(b64)

    try:
        messages = [
            {"role":"system","content":"Extract short product keywords from the image (brand, category, style/color). Return a comma-separated list; no sentences."},
            {"role":"user","content":[
                {"type":"text","text":"Give concise keywords only."},
                {"type":"image_url","image_url":{"url": data_uri}}
            ]}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        kw = resp.choices[0].message.content or ""
        return " ".join(_toks(kw))
    except Exception as e:
        print("Vision extract error:", e)
        return ""
    



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

# ------------ Chat (text or text+image, with LLM intent routing) ------------
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

    # --- Decide intent with LLM: RECOMMEND_TEXT / RECOMMEND_IMAGE / CHAT_TEXT / CHAT_IMAGE
    intent = classify_intent_llm(user_msg, bool(image_url), image_url=image_url)

    # ---------- Branch 1: recommendation via IMAGE (image-only or image+text)
    if intent == "RECOMMEND_IMAGE":
        # turn the uploaded image into compact keywords; combine with any user text
        q_img = image_to_query(image_url) if image_url else ""
        query = f"{user_msg} {q_img}".strip() if user_msg else (q_img or "product")

        items = search_catalog_semantic(user_msg, topk=5)

        reply = f"(query: {query})\n" + format_products_text(items)

        # store user turn (image and optional text) in history
        if image_url:
            # build the same multi-part content shape you already use
            if image_url.startswith("/"):
                img_path = UPLOAD_DIR / pathlib.Path(image_url).name
            else:
                img_path = ROOT / pathlib.Path(image_url)
            encoded_image = encode_image(img_path)
            parts = []
            if user_msg:
                parts.append({"type": "text", "text": user_msg})
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
            hist.append({"role": "user", "content": parts})
        else:
            hist.append({"role": "user", "content": user_msg or "(image)"})

        hist.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "session_id": session_id})

    # ---------- Branch 2: recommendation via TEXT
    if intent == "RECOMMEND_TEXT":
        items = search_catalog_semantic(user_msg, topk=5)
        reply = format_products_text(items)

        hist.append({"role": "user", "content": user_msg})
        hist.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "session_id": session_id})

    # ---------- Branch 3: general chat (text or image)
    # Build OpenAI messages (keep your existing logic)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(hist))

    if image_url:
        # multi-part content (your current format)
        if image_url.startswith("/"):
            img_path = UPLOAD_DIR / pathlib.Path(image_url).name
        else:
            img_path = ROOT / pathlib.Path(image_url)
        encoded_image = encode_image(img_path)

        user_content = []
        if user_msg:
            user_content.append({"type": "text", "text": user_msg})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_msg})

    # Call model for normal chat
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
    )
    reply = resp.choices[0].message.content

    # Update memory (unchanged from your original)
    if image_url:
        hist.append({"role": "user", "content": user_content})
    else:
        hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "session_id": session_id})

# ------------ Catalog ------------
@app.route("/catalog", methods=["GET"])
def get_catalog():
    """
    Debug endpoint: return indexed catalog (keywords included).
    """
    return jsonify(CATALOG)


# ------------ Dev Server ------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)