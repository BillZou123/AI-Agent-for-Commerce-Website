import base64
import string
import re

ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}

def allowed_file(filename):
    """Validate extension against a small allowlist."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def encode_image(image_path):
    """Return base64 (utf-8) of an image file path."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def toks(s):
    # lowercase + only keep alphanumeric tokens
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def norm(s):
    """Simple normalization for tags/brands/categories."""
    return (s or "").strip().lower()