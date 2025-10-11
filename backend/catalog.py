import json
from .config import CATALOG_JSON
from .utils import toks, norm

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

def format_products_text(items):
    """
    Render a numbered, readable block the frontend parser understands.
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