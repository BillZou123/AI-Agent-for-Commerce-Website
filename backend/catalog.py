import json
from .config import CATALOG_JSON
from .utils import toks, norm
import os, math
from openai import OpenAI

# Lazy OpenAI client for embeddings
_DEF_EMB_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_client = None
def _client_or_none():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        _client = OpenAI(api_key=api_key)
    return _client

# --- Simple vector helpers (no numpy) ---
def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def _norm(a):
    return math.sqrt(sum(x*x for x in a)) or 1.0

def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))

# --- Caches ---
_TAG_EMBED_CACHE = {}   # tag -> [float,...]
_TOKEN_EMBED_CACHE = {} # token -> [float,...]

def _embed_texts(texts, model=None):
    """
    Embed a list of strings; returns a list of embeddings.
    Falls back to [] on error.
    """
    cli = _client_or_none()
    if not cli:
        return []
    model = model or _DEF_EMB_MODEL
    # Render may batch aggressively; keep batches modest
    out = []
    B = 96
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        try:
            res = cli.embeddings.create(model=model, input=chunk)
            # API returns objects with .embedding for each data item
            out.extend([d.embedding for d in res.data])
        except Exception as e:
            print("embedding error:", e)
            # keep alignment with chunk length (skip)
            out.extend([[] for _ in chunk])
    return out

def _ensure_tag_embeddings():
    """
    Precompute embeddings for all tags in TAG_VOCAB if not cached.
    """
    missing = [t for t in TAG_VOCAB if t not in _TAG_EMBED_CACHE]
    if not missing:
        return
    vecs = _embed_texts(missing)
    if not vecs:
        return
    for t, v in zip(missing, vecs):
        _TAG_EMBED_CACHE[t] = v

def _token_embedding(tok):
    """
    Get (and cache) the embedding for a single token.
    """
    tok = norm(tok)
    if tok in _TOKEN_EMBED_CACHE:
        return _TOKEN_EMBED_CACHE[tok]
    vecs = _embed_texts([tok])
    if not vecs or not vecs[0]:
        return None
    _TOKEN_EMBED_CACHE[tok] = vecs[0]
    return vecs[0]

def search_catalog_embeddings(query, topk=5, threshold=0.85, max_tags_per_token=5):
    """
    Embedding-based semantic tag search.
    1) tokenize the query
    2) embed each token
    3) compare to tag embeddings; select tags with cosine >= threshold
    4) score items by overlap with selected tags (+ brand/category boosts)
    5) fallback to keyword search on miss or error
    """
    terms = toks(query)
    if not terms:
        return search_catalog(query, topk=topk)

    # Ensure we have tag embeddings
    _ensure_tag_embeddings()
    if not _TAG_EMBED_CACHE:  # no embeddings available (likely missing API key)
        return search_catalog(query, topk=topk)

    # Build a set of selected tags via similarity
    selected_tags = set()
    tag_items = list(_TAG_EMBED_CACHE.items())  # [(tag, vec), ...]

    for t in terms:
        tv = _token_embedding(t)
        if tv is None:
            continue
        # rank tags by cosine similarity to token embedding
        sims = []
        for tag, vec in tag_items:
            if not vec:
                continue
            sims.append(( _cos(tv, vec), tag ))
        sims.sort(reverse=True, key=lambda x: x[0])

        # pick tags above threshold, up to max_tags_per_token
        picked = 0
        for sim, tag in sims:
            if sim < threshold:
                break
            selected_tags.add(tag)
            picked += 1
            if picked >= max_tags_per_token:
                break

    if not selected_tags:
        # if nothing matched semantically, fall back
        return search_catalog(query, topk=topk)

    # Score products by overlap with selected tags (+ brand/category boosts)
    ranked = []
    term_set = set(terms)
    for it in CATALOG:
        item_tags = set(norm(t) for t in it.get("tags", []))
        score = len(selected_tags & item_tags)

        brand = norm(it.get("brand"))
        cat   = norm(it.get("category"))
        if brand and brand in term_set: score += 2
        if cat   and cat   in term_set: score += 1

        if score > 0:
            ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    hits = [it for _, it in ranked[:int(topk)]]

    # Fallback if still empty
    if not hits:
        return search_catalog(query, topk=topk)
    return hits

# Optional: keep a friendly alias so callers can switch with minimal changes
def search_catalog_llm(query, topk=5, threshold=0.85):
    """
    Convenience wrapper that uses embedding-based semantic matching.
    """
    return search_catalog_embeddings(query, topk=topk, threshold=threshold)

# ---- Load catalog and build derived fields ----
with open(CATALOG_JSON, "r") as f:
    RAW_CATALOG = json.load(f)

CATALOG = []
for item in RAW_CATALOG:
    # Build a lightweight keyword set from name/category/brand/tags
    kw = set(toks(item.get("name"))) \
       | set(toks(item.get("category"))) \
       | set(toks(item.get("brand"))) \
       | set().union(*[set(toks(t)) for t in item.get("tags", [])])
    item["keywords"] = sorted(list(kw))
    CATALOG.append(item)

# Tag vocabulary (normalized) for LLM mapping
TAG_VOCAB = sorted(set(
    norm(tag)
    for it in CATALOG
    for tag in it.get("tags", []) if tag is not None
))


def search_catalog(query, topk=5):
    """
    Token-overlap search over prebuilt keywords with small boosts.
    """
    q = set(toks(query))
    ranked = []
    for it in CATALOG:
        kw = set(it.get("keywords", []))
        score = len(q & kw)
        brand = (it.get("brand") or "").lower()
        cat   = (it.get("category") or "").lower()
        if brand in q: score += 2
        if cat   in q: score += 1
        if score > 0:
            ranked.append((score, it))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in ranked[:int(topk)]]

'''
def search_catalog_semantic(query, tag_mapper, topk=5):
    """
    LLM-assisted tag search:
      1) tokenize query
      2) map each term -> allowed catalog tags via LLM (tag_mapper)
      3) score items by tag overlap (+ small brand/category boosts)
      4) fallback to token-overlap if nothing matched
    """
    terms = toks(query)
    tag_candidates = set()
    for t in terms:
        tag_candidates |= tag_mapper(t)   # returns a set of tags

    ranked = []
    for it in CATALOG:
        item_tags = set(norm(t) for t in it.get("tags", []))
        score = len(tag_candidates & item_tags)

        brand = norm(it.get("brand"))
        cat   = norm(it.get("category"))
        if brand and brand in terms: score += 2
        if cat   and cat   in terms: score += 1

        if score > 0:
            ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    hits = [it for _, it in ranked[:int(topk)]]

    if not hits:
        hits = search_catalog(query, topk=topk)
    return hits

'''

def format_products_text(items, header=None):
    """
    Render a numbered, readable block the frontend parser understands.
    """
    if not items:
        return "Sorry, I couldn’t find a match in my catalog."

    # default header (text-based search)
    header = header or "Here are some picks from my catalog:"

    lines = [header]
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
        lines.append("")  # blank line
    return "\n".join(lines).rstrip()
 