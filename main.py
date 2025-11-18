import os
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


# ---------- Marketplace search helpers ----------

class SearchResult(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    thumbnail: Optional[str] = None
    price: Optional[str] = None


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
                    source="Pamono",
                    title=it.get("title"),
                    url=it.get("link"),
                    snippet=it.get("snippet"),
                    thumbnail=cse_thumb,
                    price=None,
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
                    source=site,
                    title=it.get("title"),
                    url=it.get("link"),
                    snippet=it.get("snippet"),
                    thumbnail=(it.get("thumbnailUrl") or it.get("imageUrl")),
                    price=None,
                )
            )
        return results
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Serper search error: {str(e)}")


def search_marketplaces(query: str) -> Dict[str, List[SearchResult]]:
    return {
        "Auctionet": serper_search(query, "auctionet.com"),
        "1stDibs": serper_search(query, "1stdibs.com"),
        "Pamono": search_pamono(query),
    }


# ---------- API models ----------

class IdentifyResponse(BaseModel):
    vision: Dict[str, Any]
    entities: Dict[str, Any]
    gpt: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    query: str
    attribution: Dict[str, Any]
    results: Dict[str, List[SearchResult]]
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
    return search_marketplaces(body.query)


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
    results = search_marketplaces(query)
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
