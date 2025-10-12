
# =============================================================================
#          DEPRECATED / NOT IN USE
# =============================================================================

import re
from openai import OpenAI
from .config import api_key, UPLOAD_DIR, ROOT
from .utils import encode_image, toks, norm
from .catalog import TAG_VOCAB

client = OpenAI(api_key=api_key)

# Cache term->tags to reduce LLM calls
_LLM_TAG_CACHE = {}  # {term: set(tags)}

# ------------------------------------------------------------
# Map a user term to catalog tags using LLM
# ------------------------------------------------------------
def map_term_to_tags_with_llm(term):
    """
    Uses LLM to map a single user-provided term into 1–5 catalog tags 
    from TAG_VOCAB. Results are cached for reuse.
    """
    term = norm(term)
    if not term:
        return set()
    if term in _LLM_TAG_CACHE:
        return _LLM_TAG_CACHE[term]

    sys = (
        "You map user terms to tags from a fixed vocabulary for product search. "
        "Return ONLY tags that appear in the provided vocabulary. "
        "Output a comma-separated list of 1-5 tags; return 'none' if nothing fits."
    )
    vocab_str = ", ".join(TAG_VOCAB[:400])
    user = f"term: {term}\nallowed_tags: {vocab_str}\nrespond with tags only."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:
        print("tag map llm error:", e)
        text = "none"

    candidates = [norm(x) for x in re.split(r"[,\n]+", text)]
    tags = set([c for c in candidates if c in TAG_VOCAB])
    _LLM_TAG_CACHE[term] = tags
    return tags

# ------------------------------------------------------------
# Classify user intent (text vs image, chat vs shopping)
# ------------------------------------------------------------
def classify_intent_llm(user_text, has_image, image_url=None):
    """
    Uses LLM to classify user intent into one of:
    RECOMMEND_TEXT, RECOMMEND_IMAGE, CHAT_TEXT, CHAT_IMAGE.
    Considers both text and optional image input.
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
        "- If the image is a chart/graph, document/screenshot, scenery, face/selfie, or anything not obviously a shopping product -> classify as CHAT_IMAGE.\n"
        "- Return ONLY the label token."
    )

    text_summary = f"has_image={str(bool(has_image)).lower()}\ntext={user_text or '(empty)'}"
    content = [{"type": "text", "text": text_summary}]
    if image_url:
        if image_url.startswith("/"):
            img_path = UPLOAD_DIR / image_url.split("/")[-1]
        else:
            img_path = ROOT / image_url
        b64 = encode_image(img_path)
        data_uri = f"data:image/jpeg;base64,{b64}"
        content.append({"type":"image_url","image_url":{"url":data_uri}})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":content}],
            temperature=0.0,
        )
        label = (resp.choices[0].message.content or "").strip().upper()
        valid = {"RECOMMEND_TEXT","RECOMMEND_IMAGE","CHAT_TEXT","CHAT_IMAGE"}
        if label not in valid:
            label = "CHAT_IMAGE" if has_image else "CHAT_TEXT"
        print("***********checking Intent label*************")
        print(label)
        return label
    except Exception as e:
        print("Intent classification error:", e)
        return "CHAT_IMAGE" if has_image else "CHAT_TEXT"

# ------------------------------------------------------------
# Extract product keywords from an image
# ------------------------------------------------------------
def image_to_query(image_url):
    """
    Uses a vision LLM to extract concise product-related keywords 
    (brand, category, style, color) from an uploaded image.
    """
    if not image_url:
        return ""

    if image_url.startswith("/"):
        img_name = image_url.split("/")[-1]
        img_path = UPLOAD_DIR / img_name
    else:
        img_path = ROOT / image_url

    b64 = encode_image(img_path)
    data_uri = f"data:image/jpeg;base64,{b64}"

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
        return " ".join(toks(kw))
    except Exception as e:
        print("Vision extract error:", e)
        return ""
    
