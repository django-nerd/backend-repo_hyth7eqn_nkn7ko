import os
import base64
import io
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env at startup
load_dotenv()

APP_NAME = "Vintage Finds API"

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Environment helpers ----------

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing environment variable: {name}")
    return value


def file_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def fetch_bytes_from_url(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")


def extract_image_from_page(url: str) -> Optional[str]:
    """Try to extract an image (og:image / twitter:image) from a generic webpage URL."""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
        # Simple regexes for meta tags
        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                return m.group(1)
    except requests.RequestException:
        return None
    return None


# ---------- Stage 1: Google Vision Web Detection ----------

def google_vision_web_detection(image_b64: str) -> Dict[str, Any]:
    api_key = require_env("EXPO_PUBLIC_GOOGLE_VISION_API_KEY")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    body = {
        "requests": [
            {
                "image": {"content": image_b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 20},
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                ],
            }
        ]
    }
    try:
        r = requests.post(url, json=body, timeout=25)
        r.raise_for_status()
        data = r.json()
        return data.get("responses", [{}])[0]
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Google Vision error: {str(e)}")


def extract_entities_from_vision(vision: Dict[str, Any]) -> Dict[str, Any]:
    web = vision.get("webDetection", {}) if vision else {}
    labels = vision.get("labelAnnotations", [])

    web_entities = [
        {"description": e.get("description"), "score": e.get("score")}
        for e in web.get("webEntities", [])
        if e.get("description")
    ]
    best_guesses = [b.get("label") for b in web.get("bestGuessLabels", []) if b.get("label")]
    pages = [
        {
            "url": p.get("url"),
            "pageTitle": p.get("pageTitle"),
        }
        for p in web.get("pagesWithMatchingImages", [])
        if p.get("url")
    ]
    full_matches = [m.get("url") for m in web.get("fullMatchingImages", []) if m.get("url")]
    partial_matches = [m.get("url") for m in web.get("partialMatchingImages", []) if m.get("url")]

    label_texts = [l.get("description") for l in labels if l.get("description")]

    # Heuristic hints likely relevant for designer/model/manufacturer
    hints = []
    for item in (best_guesses + label_texts + [e["description"] for e in web_entities]):
        if not item:
            continue
        if any(k in item.lower() for k in ["model", "no.", "no", "nr", "n°", "table", "chair", "sofa", "lamp", "pendant", "stool", "sideboard", "coffee", "dining", "armchair", "bureau", "desk", "dresser", "cabinet", "sconce", "chandelier", "vase", "candle"]):
            hints.append(item)
    hints = list(dict.fromkeys(hints))  # de-duplicate, keep order

    return {
        "webEntities": web_entities,
        "bestGuesses": best_guesses,
        "pages": pages,
        "fullMatches": full_matches,
        "partialMatches": partial_matches,
        "labels": label_texts,
        "hints": hints,
    }


# ---------- Stage 2: OpenAI GPT-4o Vision ----------

def openai_generate_query(image_b64: str, vision_entities: Dict[str, Any], user_description: Optional[str] = None) -> Dict[str, Any]:
    api_key = require_env("EXPO_PUBLIC_VIBECODE_OPENAI_API_KEY")

    system_prompt = (
        "You are an expert in mid-century and vintage furniture attribution. "
        "You receive a furniture photo and web hints (designer names, model numbers, manufacturers, brand, style) "
        "from Google Vision Web Detection. Combine visual reasoning with the hints to produce: "
        "1) a precise, marketplace-optimized search query, and 2) a structured attribution. "
        "Prefer specificity over generic style terms. If uncertain, pick the highest-probability attribution. "
        "Output strict JSON with keys: query, designer, manufacturer, model, year, style, confidence (0-1), notes."
    )

    user_instructions = (
        "Generate a single best query suitable for Auctionet, 1stDibs, and Pamono. "
        "Query should look like: 'Designer Model Name Manufacturer' with type (e.g., coffee table, pendant lamp) if clear. "
        "Avoid extra punctuation or superfluous adjectives."
    )

    # Build content with image as data URL
    data_url = f"data:image/jpeg;base64,{image_b64}"

    content: List[Any] = [
        {"type": "text", "text": user_instructions},
        {"type": "text", "text": "Web hints:"},
        {"type": "text", "text": json.dumps(vision_entities, ensure_ascii=False)},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    if user_description:
        content.insert(0, {"type": "text", "text": f"User description: {user_description}"})

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        content = choice["message"]["content"]
        result = json.loads(content)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {str(e)}")
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"OpenAI parse error: {str(e)}")

    # Normalize fields
    return {
        "query": result.get("query"),
        "designer": result.get("designer"),
        "manufacturer": result.get("manufacturer"),
        "model": result.get("model"),
        "year": result.get("year"),
        "style": result.get("style"),
        "confidence": result.get("confidence"),
        "notes": result.get("notes"),
    }


# ---------- Marketplace post-processing rules ----------

NON_PRODUCT_SEGMENTS = [
    "/blog",
    "/editorial",
    "/articles",
    "/stories",
    "/inspiration",
    "/search",
    "/shop",
    "/collections",
    "/categories",
    "/results",
]

SOLD_KEYWORDS = [
    "slutpris",
    "såld",
    "sold",
    "/sold/",
    "final price",
    "closing price",
    "avslutad",
    "closed",
    "ended",
    "resultat",
    "lot closed",
    "auktionen är avslutad",
]

ACTIVE_AUCTION_MARKERS = [
    "dagen",
    "pågående",
    "auktionerar",
    "current bid",
    "bidding is open",
    "auction ends in",
]

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp)(\?.*)?$", re.IGNORECASE)


def is_non_product_url(url: str) -> bool:
    u = url.lower()
    return any(seg in u for seg in NON_PRODUCT_SEGMENTS)


def strip_tracking(url: str) -> str:
    # Remove common tracking parameters
    if "?" not in url:
        return url
    base, qs = url.split("?", 1)
    clean_params = []
    for pair in qs.split("&"):
        key = pair.split("=", 1)[0].lower()
        if key in ("utm_source", "utm_medium", "utm_campaign", "referrer", "analytics", "utm_term", "utm_content"):
            continue
        clean_params.append(pair)
    if not clean_params:
        return base
    return base + "?" + "&".join(clean_params)


def pick_image_from_item(item: Dict[str, Any]) -> Optional[str]:
    # Check common fields from Serper and CSE
    # Order specified by requirements
    candidates: List[Optional[str]] = []
    candidates.append(item.get("thumbnailUrl"))
    candidates.append(item.get("image") or item.get("imageUrl"))
    imgs = item.get("images")
    if isinstance(imgs, list) and imgs:
        candidates.append(imgs[0] if isinstance(imgs[0], str) else imgs[0].get("url"))
    pagemap = item.get("pagemap") or {}
    cse_image = pagemap.get("cse_image")
    if isinstance(cse_image, list) and cse_image:
        candidates.append(cse_image[0].get("src"))
    metatags_list = pagemap.get("metatags")
    metatags = metatags_list[0] if isinstance(metatags_list, list) and metatags_list else {}
    if metatags:
        candidates.append(metatags.get("og:image"))
        candidates.append(metatags.get("twitter:image"))
    cse_thumb = pagemap.get("cse_thumbnail")
    if isinstance(cse_thumb, list) and cse_thumb:
        candidates.append(cse_thumb[0].get("src"))

    for c in candidates:
        if c and isinstance(c, str):
            return normalize_image_url(c)
    return None


def normalize_image_url(url: str) -> str:
    # If no conventional image extension, append safe fallback
    if IMG_EXT_RE.search(url):
        return url
    # keep existing query if present
    return url + ("&" if "?" in url else "?") + "format=jpg"


CURRENCY_MAP = {
    "€": "EUR",
    "eur": "EUR",
    "sek": "SEK",
    "kr": "SEK",
    "usd": "USD",
    "$": "USD",
    "gbp": "GBP",
    "£": "GBP",
}


def normalize_price(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    # Try to find currency symbol or code and number
    m = re.search(r"(€|\$|£|sek|eur|usd|gbp|kr)\s*([0-9][0-9\s.,]*)", t, re.IGNORECASE)
    if not m:
        m = re.search(r"([0-9][0-9\s.,]*)\s*(€|\$|£|sek|eur|usd|gbp|kr)", t, re.IGNORECASE)
    if not m:
        return None
    g1, g2 = m.group(1), m.group(2) if m.lastindex and m.lastindex >= 2 else None
    if g2 is None:
        # first pattern matched
        currency_raw, num_raw = g1, m.group(2)
    else:
        # second pattern matched
        num_raw, currency_raw = g1, g2
    currency = CURRENCY_MAP.get(currency_raw.lower(), None)
    # Sanitize numeric string
    num = re.sub(r"[\s,]", "", num_raw)
    try:
        value = float(num)
    except ValueError:
        # try replacing comma as decimal separator
        try:
            value = float(num.replace(".", "").replace(",", "."))
        except ValueError:
            return None
    return {"value": value, "currency": currency}


def contains_any(text: str, keywords: List[str]) -> bool:
    lt = text.lower()
    return any(k in lt for k in keywords)


class MarketplaceItem(BaseModel):
    source: str
    title: str = ""
    url: str
    image: str
    price: Dict[str, Optional[Any]] = Field(default_factory=lambda: {"value": None, "currency": None})
    location: str = ""
    isActive: bool = True
    relevanceScore: int = 0


# ---------- Marketplace search helpers ----------

class SearchResult(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    thumbnail: Optional[str] = None
    price: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def search_pamono(query: str) -> List[SearchResult]:
    api_key = require_env("EXPO_PUBLIC_GOOGLE_SEARCH_API_KEY")
    cx = require_env("EXPO_PUBLIC_PAMONO_SEARCH_ENGINE_ID")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        results: List[SearchResult] = []
        for it in items:
            pagemap = it.get("pagemap", {})
            cse_thumb = None
            if pagemap.get("cse_thumbnail"):
                cse_thumb = pagemap["cse_thumbnail"][0].get("src")
            results.append(
                SearchResult(
                    source="pamono",
                    title=it.get("title"),
                    url=it.get("link"),
                    snippet=it.get("snippet"),
                    thumbnail=cse_thumb,
                    price=None,
                    raw=it,
                )
            )
        return results
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Pamono search error: {str(e)}")


def serper_search(query: str, site: str) -> List[SearchResult]:
    api_key = require_env("EXPO_PUBLIC_SERPER_API_KEY")
    url = "https://google.serper.dev/search"
    body = {"q": f"site:{site} {query}", "num": 10}
    try:
        r = requests.post(url, headers={"X-API-KEY": api_key, "Content-Type": "application/json"}, json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = data.get("organic", [])
        results: List[SearchResult] = []
        for it in items:
            results.append(
                SearchResult(
                    source=site.split(".")[0].lower(),
                    title=it.get("title"),
                    url=it.get("link"),
                    snippet=it.get("snippet"),
                    thumbnail=(it.get("thumbnailUrl") or it.get("imageUrl")),
                    price=None,
                    raw=it,
                )
            )
        return results
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Serper search error: {str(e)}")


def _is_1stdibs_product(raw: Dict[str, Any], url: str) -> bool:
    # Prefer explicit signals from page metadata
    pagemap = raw.get("pagemap") or {}
    metatags_list = pagemap.get("metatags")
    metatags = metatags_list[0] if isinstance(metatags_list, list) and metatags_list else {}
    og_type = (metatags.get("og:type") or metatags.get("og:type:content") or "").lower()
    return ("/id-" in url) or (og_type == "product")


def enforce_rules_and_normalize(items: List[SearchResult], source: str, query_terms: Dict[str, str], strict: bool = True) -> List[MarketplaceItem]:
    normalized: List[MarketplaceItem] = []
    seen_urls: Set[str] = set()
    for r in items:
        url = (r.url or "").strip()
        title = r.title or ""
        snippet = r.snippet or ""
        raw = r.raw or {}
        src = source

        if not url:
            continue

        # Global URL filters
        if is_non_product_url(url):
            continue

        # Source-specific filters
        if src == "auctionet":
            # reject sold/closed
            if contains_any(url, SOLD_KEYWORDS) or contains_any(title, SOLD_KEYWORDS) or contains_any(snippet, SOLD_KEYWORDS):
                continue
            if strict:
                # require active markers only in strict mode
                is_active_marker = contains_any(title, ACTIVE_AUCTION_MARKERS) or contains_any(snippet, ACTIVE_AUCTION_MARKERS)
                if not is_active_marker:
                    continue
        elif src == "1stdibs":
            # reject known non-product sections
            if contains_any(url, ["/story/", "/editorial/", "/article/", "/our-edit/", "/collections", "/shop/", "/results"]):
                continue
            if strict:
                # require /id- or explicit product meta in strict mode
                if not _is_1stdibs_product(raw, url):
                    continue
            # always strip tracking
            url = strip_tracking(url)
        elif src == "pamono":
            # Pamono product pages contain /p/
            if "/p/" not in url:
                continue

        # Image selection
        image_url = r.thumbnail or pick_image_from_item(raw)
        if not image_url:
            continue
        image_url = normalize_image_url(image_url)

        # Price normalization (from snippet/title if present)
        price_struct = normalize_price(r.price or snippet or title) or {"value": None, "currency": None}

        # Location heuristic (from snippet)
        location = ""
        mloc = re.search(r"(located in|ships from)\s+([^.|,]+)", snippet, re.IGNORECASE)
        if mloc:
            location = mloc.group(2).strip()

        # Relevance scoring
        score = 0
        q = query_terms
        low_title = (title or "").lower()
        if q.get("designer") and q["designer"].lower() in low_title:
            score += 15
        if q.get("model") and q["model"].lower() in low_title:
            score += 10
        if q.get("manufacturer") and q["manufacturer"].lower() in low_title:
            score += 8
        if q.get("material") and q["material"].lower() in low_title:
            score += 5
        if q.get("style") and q["style"].lower() in low_title:
            score += 3
        if q.get("category") and q["category"].lower() in low_title:
            score += 2

        # Auctionet bonus if ends > 6h (heuristic via snippet like "X hours")
        if src == "auctionet":
            mh = re.search(r"(\d+)\s+hours?", snippet, re.IGNORECASE)
            if mh and int(mh.group(1)) > 6:
                score += 5

        # De-duplicate by URL
        if url in seen_urls:
            continue
        seen_urls.add(url)

        normalized.append(
            MarketplaceItem(
                source=src,
                title=title,
                url=url,
                image=image_url,
                price=price_struct,
                location=location,
                isActive=True,
                relevanceScore=score,
            )
        )
    return normalized


def _take_top(items: List[MarketplaceItem], n: int = 4) -> List[MarketplaceItem]:
    # Sort by relevance desc, then stable title tie-breaker
    items_sorted = sorted(items, key=lambda x: (-x.relevanceScore, x.title or ""))
    return items_sorted[:n]


def search_marketplaces_structured(query: str, query_terms: Optional[Dict[str, str]] = None) -> Dict[str, List[MarketplaceItem]]:
    # derive site-specific searches
    auctionet_raw = serper_search(query, "auctionet.com")
    firstdibs_raw = serper_search(query, "1stdibs.com")
    pamono_raw = search_pamono(query)

    q_terms = query_terms or {}

    # Strict pass
    auctionet_items = enforce_rules_and_normalize(auctionet_raw, "auctionet", q_terms, strict=True)
    firstdibs_items = enforce_rules_and_normalize(firstdibs_raw, "1stdibs", q_terms, strict=True)
    pamono_items = enforce_rules_and_normalize(pamono_raw, "pamono", q_terms, strict=True)

    # Relaxed fallback if fewer than 4
    if len(auctionet_items) < 4:
        relaxed = enforce_rules_and_normalize(auctionet_raw, "auctionet", q_terms, strict=False)
        # merge unique
        urls = {i.url for i in auctionet_items}
        for it in relaxed:
            if it.url not in urls:
                auctionet_items.append(it)
                urls.add(it.url)

    if len(firstdibs_items) < 4:
        relaxed = enforce_rules_and_normalize(firstdibs_raw, "1stdibs", q_terms, strict=False)
        urls = {i.url for i in firstdibs_items}
        for it in relaxed:
            if it.url not in urls:
                firstdibs_items.append(it)
                urls.add(it.url)

    if len(pamono_items) < 4:
        # for Pamono, relaxed == strict (rule stays /p/), but we still re-run in case of new images/prices
        relaxed = enforce_rules_and_normalize(pamono_raw, "pamono", q_terms, strict=False)
        urls = {i.url for i in pamono_items}
        for it in relaxed:
            if it.url not in urls:
                pamono_items.append(it)
                urls.add(it.url)

    # Cap to 4 and sort by relevance
    return {
        "Auctionet": _take_top(auctionet_items, 4),
        "1stDibs": _take_top(firstdibs_items, 4),
        "Pamono": _take_top(pamono_items, 4),
    }


# ---------- API models ----------

class IdentifyResponse(BaseModel):
    vision: Dict[str, Any]
    entities: Dict[str, Any]
    gpt: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    query: str
    attribution: Dict[str, Any]
    results: Dict[str, List[MarketplaceItem]]
    raw: IdentifyResponse


# ---------- Routes ----------

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "ok"}


@app.get("/test")
def test():
    return {
        "backend": "ok",
        "env": {
            "OPENAI": bool(os.getenv("EXPO_PUBLIC_VIBECODE_OPENAI_API_KEY")),
            "GOOGLE_VISION": bool(os.getenv("EXPO_PUBLIC_GOOGLE_VISION_API_KEY")),
            "GOOGLE_SEARCH": bool(os.getenv("EXPO_PUBLIC_GOOGLE_SEARCH_API_KEY")),
            "PAMONO_CX": bool(os.getenv("EXPO_PUBLIC_PAMONO_SEARCH_ENGINE_ID")),
            "SERPER": bool(os.getenv("EXPO_PUBLIC_SERPER_API_KEY")),
        },
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: Optional[UploadFile] = File(None),
    original_filename: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    try:
        content: Optional[bytes] = None
        if file is not None:
            content = await file.read()
        elif image_url:
            # If it's likely a webpage, try to extract og:image first
            if not re.search(r"\.(jpg|jpeg|png|webp|gif)(\?.*)?$", image_url, re.IGNORECASE):
                extracted = extract_image_from_page(image_url)
                if extracted:
                    image_url = extracted
            content = fetch_bytes_from_url(image_url)
        if not content:
            raise HTTPException(status_code=400, detail="No image provided")

        image_b64 = file_to_base64(content)

        # Stage 1: Google Vision
        vision_raw = google_vision_web_detection(image_b64)
        entities = extract_entities_from_vision(vision_raw)

        # Stage 2: OpenAI vision
        gpt = openai_generate_query(image_b64, entities, user_description=description)

        return IdentifyResponse(vision=vision_raw, entities=entities, gpt=gpt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchBody(BaseModel):
    query: str


@app.post("/search")
def search(body: SearchBody):
    # Use query terms heuristics from the query itself
    q = body.query
    terms = {
        "designer": "",
        "model": "",
        "manufacturer": "",
        "material": "",
        "style": "",
        "category": "",
    }
    # simple heuristics (e.g., words like 'lamp', 'table' as category)
    for cat in ["lamp", "table", "chair", "sofa", "pendant", "sideboard", "desk", "stool", "sconce", "chandelier"]:
        if cat in q.lower():
            terms["category"] = cat
            break
    return search_marketplaces_structured(body.query, terms)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    ident = await identify(file=file, image_url=image_url, description=description)
    query = ident.gpt.get("query") or ""
    if not query:
        raise HTTPException(status_code=500, detail="Failed to generate query")

    # Build query terms from attribution to score relevance
    q_terms = {
        "designer": ident.gpt.get("designer") or "",
        "manufacturer": ident.gpt.get("manufacturer") or "",
        "model": ident.gpt.get("model") or "",
        "style": ident.gpt.get("style") or "",
        # Materials not provided explicitly; try hints
        "material": (next((h for h in ident.entities.get("labels", []) if h and h.lower() in [
            "brass", "teak", "oak", "rosewood", "steel", "aluminum", "leather", "glass", "marble"
        ]), "")),
        # Category from hints/best guesses
        "category": (next((h for h in (ident.entities.get("hints", []) or []) if h and h.lower() in [
            "lamp", "table", "chair", "sofa", "pendant", "sideboard", "desk", "stool", "sconce", "chandelier", "armchair", "coffee table", "dining table"
        ]), "")),
    }

    results = search_marketplaces_structured(query, q_terms)

    attribution = {
        "designer": ident.gpt.get("designer"),
        "manufacturer": ident.gpt.get("manufacturer"),
        "model": ident.gpt.get("model"),
        "year": ident.gpt.get("year"),
        "style": ident.gpt.get("style"),
        "confidence": ident.gpt.get("confidence"),
        "notes": ident.gpt.get("notes"),
        "hints": ident.entities.get("hints"),
        "bestGuesses": ident.entities.get("bestGuesses"),
    }
    return AnalyzeResponse(query=query, attribution=attribution, results=results, raw=ident)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
