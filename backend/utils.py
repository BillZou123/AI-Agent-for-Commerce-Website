import base64
import string
import re

# ---------- File / Data Utilities ----------
# Small collection of helpers for file validation,
# image encoding, and text normalization.

# Allowed image file extensions for uploads
ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}

def allowed_file(filename):
    """
    Check if a filename has an allowed image extension.
    Returns True/False.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def encode_image(image_path):
    """
    Convert an image file into a base64-encoded string (utf-8).
    Used for embedding images directly into JSON payloads.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def toks(s):
    """
    Tokenize a string into lowercase alphanumeric tokens.
    Example: "Nike Air Zoom" -> ["nike","air","zoom"]
    """
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def norm(s):
    """
    Normalize a string to lowercase and trim whitespace.
    Useful for tags, brands, categories.
    """
    return (s or "").strip().lower()