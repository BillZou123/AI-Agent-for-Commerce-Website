import json
from .config import CATALOG_JSON
from .utils import toks, norm
import os, math
from openai import OpenAI

# ----- Embedding client setup -----
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

# ----- Vector math helpers -----
def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _norm(a): return math.sqrt(sum(x*x for x in a)) or 1.0
def _cos(a, b): return _dot(a, b) / (_norm(a) * _norm(b))

# ----- Embedding caches -----
_TAG_EMBED_CACHE = {}   # tag -> vector
_TOKEN_EMBED_CACHE = {} # token -> vector

def _embed_texts(texts, model=None):
    """Batch embed a list of strings; return vectors or [] on error."""
    cli = _client_or_none()
    if not cli: return []
    model = model or _DEF_EMB_MODEL
    out, B = [], 96
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        try:
            res = cli.embeddings.create(model=model, input=chunk)
            out.extend([d.embedding for d in res.data])
        except Exception as e:
            print("embedding error:", e)
            out.extend([[] for _ in chunk])
    return out

def _ensure_tag_embeddings():
    """Precompute embeddings for all catalog tags if missing."""
    missing = [t for t in TAG_VOCAB if t not in _TAG_EMBED_CACHE]
    if not missing: return
    vecs = _embed_texts(missing)
    if not vecs: return
    for t, v in zip(missing, vecs):
        _TAG_EMBED_CACHE[t] = v

def _token_embedding(tok):
    """Get embedding for a single token (cached)."""
    tok = norm(tok)
    if tok in _TOKEN_EMBED_CACHE:
        return _TOKEN_EMBED_CACHE[tok]
    vecs = _embed_texts([tok])
    if not vecs or not vecs[0]: return None
    _TOKEN_EMBED_CACHE[tok] = vecs[0]
    return vecs[0]

# ----- Embedding-based search -----
def search_catalog_embeddings(query, topk=5, threshold=0.85, max_tags_per_token=5):
    """
    Semantic search using embeddings:
    - tokenize query
    - embed tokens
    - match tokens to catalog tags above threshold
    - score products by tag/brand/category overlap
    - fallback to keyword search if no matches
    """
    terms = toks(query)
    if not terms: return search_catalog(query, topk=topk)

    _ensure_tag_embeddings()
    if not _TAG_EMBED_CACHE:
        return search_catalog(query, topk=topk)

    selected_tags = set()
    tag_items = list(_TAG_EMBED_CACHE.items())

    for t in terms:
        tv = _token_embedding(t)
        if tv is None: continue
        sims = [( _cos(tv, vec), tag ) for tag, vec in tag_items if vec]
        sims.sort(reverse=True, key=lambda x: x[0])

        picked = 0
        for sim, tag in sims:
            if sim < threshold: break
            selected_tags.add(tag)
            picked += 1
            if picked >= max_tags_per_token: break

    if not selected_tags:
        return search_catalog(query, topk=topk)

    ranked, term_set = [], set(terms)
    for it in CATALOG:
        item_tags = set(norm(t) for t in it.get("tags", []))
        score = len(selected_tags & item_tags)

        brand, cat = norm(it.get("brand")), norm(it.get("category"))
        if brand and brand in term_set: score += 2
        if cat   and cat   in term_set: score += 1
        if score > 0:
            ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    hits = [it for _, it in ranked[:int(topk)]]
    return hits or search_catalog(query, topk=topk)

def search_catalog_llm(query, topk=5, threshold=0.85):
    """Convenience wrapper for embedding-based search."""
    return search_catalog_embeddings(query, topk=topk, threshold=threshold)

# ----- Catalog data loading -----
with open(CATALOG_JSON, "r") as f:
    RAW_CATALOG = json.load(f)

CATALOG = []
for item in RAW_CATALOG:
    kw = set(toks(item.get("name"))) \
       | set(toks(item.get("category"))) \
       | set(toks(item.get("brand"))) \
       | set().union(*[set(toks(t)) for t in item.get("tags", [])])
    item["keywords"] = sorted(list(kw))
    CATALOG.append(item)

TAG_VOCAB = sorted(set(
    norm(tag) for it in CATALOG for tag in it.get("tags", []) if tag is not None
))

# ----- Keyword-based search -----
def search_catalog(query, topk=5):
    """Fallback keyword search with small brand/category boosts."""
    q = set(toks(query))
    ranked = []
    for it in CATALOG:
        kw = set(it.get("keywords", []))
        score = len(q & kw)
        brand, cat = (it.get("brand") or "").lower(), (it.get("category") or "").lower()
        if brand in q: score += 2
        if cat   in q: score += 1
        if score > 0: ranked.append((score, it))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in ranked[:int(topk)]]

# ----- Formatting for frontend -----
def format_products_text(items, header=None):
    """Render catalog items in numbered text block for frontend display."""
    if not items:
        return "Sorry, I couldn’t find a match in my catalog."

    header = header or "Here are some picks from my catalog:"
    lines = [header]
    for idx, it in enumerate(items, 1):
        name, price = it.get("name", "Unknown"), it.get("price")
        brand, img  = (it.get("brand") or "").title(), (it.get("image") or "")
        if price is not None:
            line = f"{idx}. **{name}** — ${price} · {brand} (image: {img})"
        else:
            line = f"{idx}. **{name}** · {brand} (image: {img})"
        lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip()

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


 