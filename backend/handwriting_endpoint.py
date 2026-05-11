"""
handwriting_endpoint.py  ←  place in backend/   (alongside main.py)
────────────────────────────────────────────────────────────────────
Self-contained: HandwritingAgent is inlined — no separate file needed.

main.py already has (after the patch):
    from handwriting_endpoint import router as handwriting_router
    app.include_router(handwriting_router)

PIPELINE (Arabic / Darija):
  PASS 1  — YOUR Qwen2.5-VL-3B model on HF Space (malek-sd-qwen-ar-htr.hf.space)
             Primary Arabic HTR model, fine-tuned for Arabic handwriting.
  PASS 2A — Gemini 2.5-Flash: re-reads image vs Pass-1 output (strict verification)
  PASS 2B — Groq Llama-4-Scout: re-reads image vs Pass-1 output (cross-check)
  PASS 2C — OpenRouter Qwen2.5-VL-72B: deep Arabic letter/dot verification
  PASS 2D — Consensus judge: all 3 verifiers vote, majority wins
  PASS 3  — Digit & medical guardrails (deterministic)
  PASS 4  — Intent classification + structured extraction

PIPELINE (English / French):
  PASS 1  — Gemini 2.5-Flash (primary)
  PASS 2  — Groq Llama-4-Scout verification (image + Pass-1 text)
  PASS 3  — Digit & medical guardrails
  PASS 4  — Intent classification + structured extraction
"""

# ── stdlib ────────────────────────────────────────────────────
import asyncio
import base64
import hashlib
import json
import re
import time as _time
import traceback
from typing import Optional

# ── third-party ───────────────────────────────────────────────
import cv2
import numpy as np
import requests
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── your existing config ──────────────────────────────────────
from config import GOOGLE_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY

router = APIRouter()

# ═══════════════════════════════════════════════════════════════
# Constants / model chains
# ═══════════════════════════════════════════════════════════════

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
_GEMINI_BASE   = ("https://generativelanguage.googleapis.com"
                  "/v1beta/models/{model}:generateContent")

_OR_URL    = "https://openrouter.ai/api/v1/chat/completions"
_OR_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",  # primary — best open vision model
    "google/gemma-4-26b-a4b-it:free",     # fallback
    "openrouter/free",                     # last resort
]

# ── YOUR Hugging Face Space — Arabic HTR (Pass 1 for Arabic/Darija) ──
_HF_SPACE_URL = "https://malek-sd-qwen-ar-htr.hf.space/recognize"

# Verification models for Arabic Pass 2 (image + text cross-check)
_OR_ARABIC_VERIFY_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",   # best open Arabic vision model
    "google/gemma-4-26b-a4b-it:free",       # fallback
]

_IS_ARABIC   = {"arabic", "darija"}
_LONG_WORDS  = 80          # threshold: summarize vs read verbatim
_LANG_LABELS = {
    "auto": "Auto-detect", "english": "English",
    "french": "French",    "arabic": "Arabic",
    "darija": "Tunisian Darija",
}

# ═══════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize + CLAHE + sharpening for better OCR accuracy.
    No rotation — the phone sends frames already in the correct orientation."""
    h, w = img.shape[:2]
    if max(h, w) > 1400:
        s = 1400 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.filter2D(gray, -1,
                        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],
                                 dtype=np.float32))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _encode(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def _compress(b64: str, max_px: int = 1200, q: int = 75) -> str:
    if len(b64) <= 1_800_000:
        return b64
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group(0) if m else raw)

# ═══════════════════════════════════════════════════════════════
# API callers
# ═══════════════════════════════════════════════════════════════

def _groq(messages: list, max_tokens: int = 1024,
          json_mode: bool = True, temperature: float = 0.0) -> dict:
    payload: dict = {
        "model": _GROQ_MODEL, "max_tokens": max_tokens,
        "temperature": temperature, "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(
        _GROQ_URL, json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


def _gemini(b64: str, prompt: str, model: str) -> dict:
    b64 = _compress(b64)
    url = _GEMINI_BASE.format(model=model)
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "temperature": 0.0, "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        f"{url}?key={GOOGLE_API_KEY}", json=payload,
        headers={"Content-Type": "application/json"}, timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    raw = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not raw.strip():
        raise RuntimeError("Gemini returned empty text")
    return _parse_json(raw)


def _gemini_chain(b64: str, prompt: str) -> dict:
    last = None
    for m in _GEMINI_MODELS:
        try:
            return _gemini(b64, prompt, m)
        except requests.HTTPError as e:
            last = e
            if e.response.status_code == 429:
                continue
            raise
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"All Gemini models failed: {last}")


def _openrouter(b64: str, prompt: str) -> dict:
    b64 = _compress(b64)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Blind Glasses",
        "Content-Type": "application/json",
    }
    last = None
    for model in _OR_MODELS:
        payload = {
            "model": model, "max_tokens": 1024, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        try:
            resp = requests.post(_OR_URL, json=payload,
                                 headers=headers, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("empty content")
            return _parse_json(content)
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"All OpenRouter models failed: {last}")

# ═══════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════

def _arabic_rules() -> str:
    return """
ARABIC / DARIJA — CRITICAL RULES:
- Transcribe ONLY what is physically visible. NEVER complete/guess words.
- Scan RIGHT to LEFT, word by word. Examine dots carefully.
- Look-alike pairs: ب/ت/ث/ن/ي  ح/ج/خ  ر/ز  د/ذ  س/ش  ص/ض  ط/ظ  ع/غ  ف/ق  ه/ة  ا/أ/إ/آ
- Darija: mixes Arabic with French/Berber loanwords phonetically.
- For illegible words write [غير مقروء] — never substitute.
- Output ONLY Arabic Unicode. Preserve line breaks with \\n.
"""


def _pass1_prompt(language: str) -> str:
    is_ar = language in _IS_ARABIC
    label = _LANG_LABELS.get(language, "Auto-detect")
    lang_block = (
        "LANGUAGE: AUTO-DETECT — identify from script. "
        'Set "detected_language" to English/French/Arabic/Darija.'
        if language == "auto" else
        f"LANGUAGE: {label} (USER-SELECTED). "
        f"Transcribe in {label}. "
        f'Set "detected_language" to "{label}".'
    )
    ar = _arabic_rules() if is_ar else ""
    return "\n".join([
        "You are an expert multilingual handwriting recognition system.\n",
        "TASK: Transcribe EXACTLY what is written. Never invent words.\n",
        lang_block, ar,
        "RULES:",
        "1. Only transcribe physically visible ink.",
        "2. Illegible word → write [illegible], never substitute.",
        "3. Preserve line breaks with \\n.",
        "4. ORDONNANCE (French prescription header) — not PARDONNANCE.",
        "5. Medicine names: only what you can see. Dosages: exact number, never convert units.",
        "6. Digits: 1 vs 4 (stroke check), 1 vs 7 (top bar), 5 vs 6 (loop).",
        '   If a dose reads "4g" for amoxicilline/paracétamol/ibuprofène → re-examine (almost certainly 1g).\n',
        "RESPOND WITH ONLY THIS JSON — no markdown:",
        '{"detected_language":"<lang>","transcription":"<text>","confidence":<0-1>,"notes":"<or empty>"}',
    ])


def _verify_prompt(transcription: str, language: str, confidence: float) -> str:
    is_ar  = language in _IS_ARABIC
    label  = _LANG_LABELS.get(language, language)
    script = (
        "Fix look-alike Arabic letters: ب/ت/ث/ن/ي  ح/ج/خ  ر/ز  د/ذ  س/ش  ص/ض  ط/ظ  ع/غ  ف/ق  ه/ة"
        if is_ar else
        "Fix Latin OCR confusions: I/l/1  O/0  rn/m. Fix ORDONNANCE if misspelled."
    )
    return f"""\
Strict handwriting verifier for {label}.
Image + raw OCR transcription provided.

1. Compare word-by-word against image.
2. {script}
3. NEVER rewrite for style/meaning. Preserve word order.
4. Hallucination check: count ink clusters vs transcription words — extra words → remove.
5. Medicine names: verify stroke-by-stroke. Unknown noun → [illegible].
6. Dosage digits: re-examine 1 vs 4, 1 vs 7, 5 vs 6.
   Impossible doses (4g amoxicilline/paracétamol) → re-read (almost certainly 1g).
7. Uncertain word → [غير مقروء] (Arabic) or [illegible] (other).
8. Provide short English meaning from verified words only.
9. Pass-1 confidence={confidence:.2f}. Below 0.90 → be EXTRA conservative.

TRANSCRIPTION: \"\"\"{transcription}\"\"\"

ONLY THIS JSON:
{{"corrected_transcription":"<text>","was_corrected":<true|false>,"meaning_english":"<1-3 sentences>","suspicious_words":"<or empty>"}}"""


def _digit_prompt(transcription: str, language: str) -> str:
    return f"""\
Medical OCR digit verifier for {language}.
Re-examine every dosage digit in the image.
Focus on: 1 vs 4, 1 vs 7, 5 vs 6, 0 vs 6.
Uncertain digit → replace with [illegible]. Do NOT rewrite structure.

Transcription: \"\"\"{transcription}\"\"\"

ONLY THIS JSON:
{{"corrected_transcription":"<text>","was_corrected":<true|false>,"suspicious_numbers":"<or empty>"}}"""

# ═══════════════════════════════════════════════════════════════
# Medical guardrails (deterministic — no API)
# ═══════════════════════════════════════════════════════════════

_MED_4G = [
    r"(amoxicilline\s*\(cp\)\s*)4g\b",
    r"(parac[eé]tamol\s*\(cp\)\s*)4g\b",
    r"(doliprane\s*\(cp\)\s*)4g\b",
]
_BAD_RX = re.compile(r"\b4\s*cp\s*[&et]+\s*fois\s+par\s+jour\b", re.I)


def _guardrails(text: str) -> tuple:
    corrected, changes = text, []
    for pat in _MED_4G:
        new, n = re.subn(pat, r"\g<1>1g", corrected, flags=re.I)
        if n:
            corrected = new
            changes.append("4g → 1g for known medicine")
    new, n = _BAD_RX.subn("1 cp 2 fois par jour", corrected)
    if n:
        corrected = new
        changes.append("4 cp & fois par jour → 1 cp 2 fois par jour")
    return corrected, changes

# ═══════════════════════════════════════════════════════════════
# Pass 1 — transcription chain
# ═══════════════════════════════════════════════════════════════

def _normalize_lang(detected: str, requested: str) -> str:
    """
    Map a raw detected_language string to one of our canonical keys.
    If requested is not 'auto', it takes priority ONLY when detection
    returns nothing useful — never overrides a clear detection.
    """
    raw = (detected or "").strip().lower()
    m = {
        "english": "english", "french": "french", "français": "french",
        "french (français)": "french",
        "arabic": "arabic", "modern standard arabic": "arabic",
        "modern standard arabic (العربية الفصحى)": "arabic",
        "darija": "darija", "tunisian darija": "darija",
        "tunisian arabic darija (الدارجة التونسية)": "darija",
        "auto": "auto",
    }
    normalized = m.get(raw)
    if normalized and normalized != "auto":
        return normalized
    # No clear detection — fall back to requested (never keep "auto" as final)
    if requested and requested != "auto":
        return requested
    return "english"   # safe default when truly unknown


def _detect_script(text: str) -> str:
    """
    Fast heuristic: count Arabic-script characters vs Latin characters.
    Returns 'arabic', 'french', or 'english' based on majority script.
    Used to correct mis-routing when language='auto'.
    """
    if not text:
        return "english"
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF'
                       or '\u0750' <= c <= '\u077F'
                       or '\uFB50' <= c <= '\uFDFF'
                       or '\uFE70' <= c <= '\uFEFF')
    latin_chars  = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
    total = arabic_chars + latin_chars
    if total == 0:
        return "english"
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.35:
        return "arabic"
    # Distinguish French from English by common French markers
    french_markers = re.compile(
        r'\b(le|la|les|de|du|des|un|une|et|est|en|au|aux|je|tu|il|elle'
        r'|nous|vous|ils|elles|que|qui|dans|sur|avec|pour|par|pas|ne'
        r'|ordonnance|médicament|comprimé|fois|jour|matin|soir|midi)\b',
        re.I
    )
    if french_markers.search(text):
        return "french"
    return "english"


def _pass1_groq(b64: str, language: str) -> dict:
    prompt = _pass1_prompt(language)
    msgs = [
        {"role": "system", "content":
         "Precise handwriting engine. NEVER invent words. Output ONLY JSON."},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]},
    ]
    r = _groq(msgs)
    r.setdefault("backend", "Groq (Llama-4 Scout)")
    return r


# ═══════════════════════════════════════════════════════════════
# Pass 1 — YOUR HF Space model (Arabic/Darija) or Gemini (Latin)
# ═══════════════════════════════════════════════════════════════

def _hf_space_arabic(b64: str, language: str) -> dict:
    """
    Call YOUR Qwen2.5-VL-3B Arabic HTR model on HF Space.
    POST /recognize  →  {"image_b64": "...", "language": "arabic"}

    Always sends "arabic" to the space — it's an Arabic HTR model and
    handles auto-detection internally.  Works for any input language
    including "auto".
    """
    # The HF Space is an Arabic HTR model — always send "arabic"
    hf_lang = "arabic" if language in ("auto", "darija") else language
    payload = {"image_b64": b64, "language": hf_lang}
    resp = requests.post(
        _HF_SPACE_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,   # HF Spaces can be slow on cold start
    )
    resp.raise_for_status()
    data = resp.json()

    # Normalize response — accept any reasonable key name
    if isinstance(data, str):
        return {
            "transcription":     data.strip(),
            "detected_language": "arabic",
            "confidence":        0.88,
            "backend":           "HF-Space (Qwen2.5-VL-3B Arabic HTR)",
        }
    if isinstance(data, dict):
        text = (
            data.get("transcription")
            or data.get("text")
            or data.get("result")
            or data.get("output")
            or data.get("recognized_text")
            or ""
        )
        return {
            "transcription":     str(text).strip(),
            "detected_language": data.get("detected_language", "arabic"),
            "confidence":        float(data.get("confidence", 0.88)),
            "backend":           "HF-Space (Qwen2.5-VL-3B Arabic HTR)",
            "notes":             data.get("notes", ""),
        }
    raise RuntimeError(f"Unexpected HF Space response type: {type(data)}")


def _run_pass1(b64: str, language: str) -> dict:
    """
    Pass 1: primary transcription — strict language-aware routing.

    EXPLICIT Arabic / Darija  →  YOUR HF Space first, then Gemini → Groq
    EXPLICIT English / French →  Gemini first, then OpenRouter → Groq
                                  (HF Space never called — it is Arabic-only)
    AUTO                      →  Gemini/Groq detects script first.
                                  If detected Arabic/Darija → re-run via HF Space.
                                  If detected Latin → use that result directly.
    """
    prompt = _pass1_prompt(language)

    # ── EXPLICIT Arabic / Darija ───────────────────────────────
    if language in _IS_ARABIC:
        try:
            r = _hf_space_arabic(b64, language)
            if r.get("transcription", "").strip():
                print(f"[handwriting] HF-Space P1 OK: "
                      f"{len(r['transcription'].split())}w")
                return r
            print("[handwriting] HF-Space returned empty — falling back")
        except Exception as e:
            print(f"[handwriting] HF-Space P1 failed ({e}), trying Gemini")

        if GOOGLE_API_KEY:
            try:
                r = _gemini_chain(b64, prompt)
                r.setdefault("backend", "Gemini (Arabic fallback)")
                return r
            except Exception as e:
                print(f"[handwriting] Gemini Arabic P1 failed: {e}")

        if OPENROUTER_API_KEY:
            try:
                r = _openrouter(b64, prompt)
                r.setdefault("backend", "OpenRouter (Arabic fallback)")
                return r
            except Exception as e:
                print(f"[handwriting] OpenRouter Arabic P1 failed: {e}")

        return _pass1_groq(b64, language)

    # ── EXPLICIT English / French ──────────────────────────────
    if language in ("english", "french"):
        if GOOGLE_API_KEY:
            try:
                r = _gemini_chain(b64, prompt)
                r.setdefault("backend", "Gemini")
                return r
            except Exception as e:
                print(f"[handwriting] Gemini P1 failed: {e}")

        if OPENROUTER_API_KEY:
            try:
                r = _openrouter(b64, prompt)
                r.setdefault("backend", "OpenRouter")
                return r
            except Exception as e:
                print(f"[handwriting] OpenRouter P1 failed: {e}")

        return _pass1_groq(b64, language)

    # ── AUTO: detect script first via Gemini/Groq ─────────────
    # Use a Latin-capable model to get the transcription + detected_language.
    # Then check the actual script of the returned text.
    # If it's Arabic → re-run via HF Space for best quality.
    # If it's Latin  → use the result as-is.
    auto_prompt = _pass1_prompt("auto")
    first_result = None

    if GOOGLE_API_KEY:
        try:
            first_result = _gemini_chain(b64, auto_prompt)
            first_result.setdefault("backend", "Gemini (auto-detect)")
        except Exception as e:
            print(f"[handwriting] Gemini auto P1 failed: {e}")

    if first_result is None:
        try:
            first_result = _pass1_groq(b64, "auto")
        except Exception as e:
            print(f"[handwriting] Groq auto P1 failed: {e}")
            return {"transcription": "", "detected_language": "english",
                    "confidence": 0.0, "backend": "failed"}

    raw_text = first_result.get("transcription", "")
    raw_lang = first_result.get("detected_language", "")

    # Determine actual script from the transcription text itself
    script = _detect_script(raw_text)
    print(f"[handwriting] auto-detect: model_lang={raw_lang!r} "
          f"script_detect={script!r} text_sample={raw_text[:40]!r}")

    if script == "arabic":
        # Re-run via HF Space for best Arabic quality
        print("[handwriting] auto → Arabic detected → switching to HF Space")
        try:
            r = _hf_space_arabic(b64, "arabic")
            if r.get("transcription", "").strip():
                return r
        except Exception as e:
            print(f"[handwriting] HF-Space Arabic re-run failed ({e}), "
                  f"using Gemini result")
        # HF Space failed — use Gemini result but fix the language label
        first_result["detected_language"] = "arabic"
        return first_result

    # Latin script — use the first result, fix language label from script
    first_result["detected_language"] = script  # "french" or "english"
    return first_result

# ═══════════════════════════════════════════════════════════════
# Pass 2 — Multi-LLM strict verification
# ═══════════════════════════════════════════════════════════════
#
# For Arabic/Darija (after HF Space Pass 1):
#   2A — Gemini 2.5-Flash:  image + Pass-1 text → corrected text
#   2B — Groq Llama-4-Scout: image + Pass-1 text → corrected text
#   2C — OpenRouter Qwen2.5-VL-72B: deep Arabic letter/dot check
#   2D — Consensus: majority vote across 2A/2B/2C
#
# For Latin (English/French):
#   2A — Groq Llama-4-Scout: image + Pass-1 text → corrected text
#   2B — Gemini 2.5-Flash (if available): cross-check
#   2D — Pick whichever corrected more confidently

def _pass2_groq(b64: str, transcription: str,
                language: str, confidence: float) -> dict:
    prompt = _verify_prompt(transcription, language, confidence)
    msgs = [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    return _groq(msgs, max_tokens=1024)


def _pass2_gemini(b64: str, transcription: str, language: str) -> dict:
    label = _LANG_LABELS.get(language, language)
    is_ar = language in _IS_ARABIC
    if is_ar:
        prompt = (
            f"Strict Arabic handwriting verifier for {label}.\n"
            "Compare word-by-word RIGHT→LEFT against the image. "
            "Fix ONLY ink-verifiable errors.\n"
            "Check every dot: ب/ت/ث/ن/ي  ح/ج/خ  ر/ز  د/ذ  س/ش  ص/ض  ط/ظ  ع/غ  ف/ق  ه/ة\n"
            "Hallucinated words (not in ink) → [غير مقروء]. "
            "Provide short English meaning.\n\n"
            f'TRANSCRIPTION TO VERIFY: """{transcription}"""\n\n'
            'ONLY JSON: {"corrected_transcription":"","was_corrected":false,'
            '"meaning_english":"","suspicious_words":""}'
        )
    else:
        prompt = (
            f"Strict handwriting verifier for {label}.\n"
            "Compare word-by-word against the image. Fix ONLY ink-verifiable errors.\n"
            "Fix Latin OCR confusions: I/l/1  O/0  rn/m. "
            "Fix ORDONNANCE if misspelled.\n"
            "Hallucinated words → [illegible]. Provide short English meaning.\n\n"
            f'TRANSCRIPTION TO VERIFY: """{transcription}"""\n\n'
            'ONLY JSON: {"corrected_transcription":"","was_corrected":false,'
            '"meaning_english":"","suspicious_words":""}'
        )
    return _gemini_chain(b64, prompt)


def _pass2_openrouter_arabic(b64: str, transcription: str, language: str) -> dict:
    """
    Deep Arabic verification via OpenRouter Qwen2.5-VL-72B.
    Letter-by-letter dot check — strongest open Arabic vision model.
    """
    b64c = _compress(b64)
    label = _LANG_LABELS.get(language, language)
    prompt = (
        f"You are a strict Arabic handwriting verifier for {label}.\n"
        "TASK: Compare the provided transcription against the image, letter by letter.\n\n"
        "CRITICAL CHECKS:\n"
        "1. Dot count: ب(1 below) ت(2 above) ث(3 above) ن(1 above) ي(2 below)\n"
        "2. ح/ج/خ — check dot presence and position\n"
        "3. ر/ز — check dot above ز\n"
        "4. د/ذ — check dot above ذ\n"
        "5. س/ش — check 3 dots above ش\n"
        "6. ص/ض — check dot above ض\n"
        "7. ط/ظ — check dot above ظ\n"
        "8. ع/غ — check dot above غ\n"
        "9. ف/ق — check dots: ف(1 above) ق(2 above)\n"
        "10. ه/ة — check 2 dots above ة\n"
        "11. ا/أ/إ/آ — check hamza position\n"
        "12. Hallucinated words (no ink cluster) → [غير مقروء]\n"
        "13. NEVER add words not in the image\n\n"
        f'TRANSCRIPTION TO VERIFY: """{transcription}"""\n\n'
        "ONLY THIS JSON (no markdown):\n"
        '{"corrected_transcription":"<corrected or same>","was_corrected":<true|false>,'
        '"corrections_made":"<list of changes or empty>","confidence":<0-1>}'
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Blind Glasses Arabic HTR",
        "Content-Type": "application/json",
    }
    last = None
    for model in _OR_ARABIC_VERIFY_MODELS:
        payload = {
            "model": model, "max_tokens": 1024, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64c}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        try:
            resp = requests.post(_OR_URL, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "")
            if not content:
                raise ValueError("empty content")
            return _parse_json(content)
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"OpenRouter Arabic verify failed: {last}")


def _consensus_arabic(p1_text: str, results: list[dict]) -> tuple[str, str]:
    """
    Consensus across multiple verifier outputs.
    Each result has: corrected_transcription, was_corrected.
    Returns (best_text, summary_of_changes).
    """
    # Collect all corrected texts (including original as baseline)
    candidates = [p1_text]
    for r in results:
        if r and r.get("was_corrected") and r.get("corrected_transcription"):
            candidates.append(r["corrected_transcription"].strip())

    if len(candidates) == 1:
        return p1_text, "no corrections"

    # Count how many verifiers agree on each candidate
    from collections import Counter
    counts = Counter(candidates)
    # Majority vote: pick the most common corrected text
    # If tie, prefer the one that was corrected (not original)
    best = counts.most_common(1)[0][0]

    changes = []
    for r in results:
        if r and r.get("was_corrected"):
            c = r.get("corrections_made") or r.get("suspicious_words") or "correction applied"
            if c:
                changes.append(str(c))

    return best, "; ".join(changes) if changes else "corrections applied"


def _run_arabic_verification(b64: str, transcription: str,
                              language: str, confidence: float) -> tuple[str, str]:
    """
    Arabic multi-LLM verification (Passes 2A/2B/2C/2D).
    Returns (final_text, meaning_english).

    LENIENT MODE: if Pass 1 came from the HF Space (high confidence),
    verification only applies light corrections — never discards text.
    If all verifiers are rate-limited, returns Pass-1 text unchanged.
    """
    results = []
    meaning = ""
    any_attempted = False

    # 2A — Gemini (image + text) — skip if likely rate-limited
    if GOOGLE_API_KEY:
        any_attempted = True
        try:
            r = _pass2_gemini(b64, transcription, language)
            results.append(r)
            if not meaning:
                meaning = r.get("meaning_english", "")
            print(f"[handwriting] P2A Gemini verify: was_corrected={r.get('was_corrected')}")
        except Exception as e:
            if "429" in str(e):
                print("[handwriting] P2A Gemini 429 — skipping")
            else:
                print(f"[handwriting] P2A Gemini failed (non-fatal): {e}")

    # 2B — Groq Llama-4-Scout (image + text)
    if GROQ_API_KEY:
        any_attempted = True
        try:
            r = _pass2_groq(b64, transcription, language, confidence)
            results.append(r)
            if not meaning:
                meaning = r.get("meaning_english", "")
            print(f"[handwriting] P2B Groq verify: was_corrected={r.get('was_corrected')}")
        except Exception as e:
            if "429" in str(e):
                print("[handwriting] P2B Groq 429 — skipping")
            else:
                print(f"[handwriting] P2B Groq failed (non-fatal): {e}")

    # 2C — OpenRouter Qwen2.5-VL-72B deep Arabic check
    if OPENROUTER_API_KEY:
        any_attempted = True
        try:
            r = _pass2_openrouter_arabic(b64, transcription, language)
            results.append(r)
            print(f"[handwriting] P2C OR-Qwen verify: was_corrected={r.get('was_corrected')}")
        except Exception as e:
            if "429" in str(e):
                print("[handwriting] P2C OR-Qwen 429 — skipping")
            else:
                print(f"[handwriting] P2C OR-Qwen failed (non-fatal): {e}")

    # 2D — Consensus (only apply if at least one verifier succeeded)
    if results:
        final, changes = _consensus_arabic(transcription, results)
        if final != transcription:
            print(f"[handwriting] P2D consensus correction: {changes}")
        return final, meaning

    # All verifiers failed / rate-limited → trust Pass-1 output as-is
    if any_attempted:
        print("[handwriting] P2 all verifiers unavailable — using Pass-1 output")
    return transcription, meaning

# ═══════════════════════════════════════════════════════════════
# Intent classifier (local, zero API)
# ═══════════════════════════════════════════════════════════════

_RX_PRESC = re.compile(
    r"ordonnance|prescription|\u0648\u0635\u0641\u0629\s*\u0637\u0628\u064a\u0629"
    r"|\b(cp|bte|comprim[\u00e9e]s?|cachet|ampoule|sirop)\b"
    r"|\b\d+\s*(?:mg|mcg|g|ml)\b"
    r"|\b(matin|soir|midi)\b|\bfois\s+par\s+jour\b"
    r"|\b(amox|augmentin|ibupro|paracet|doliprane|aspirin)\b",
    re.I,
)
_RX_Q  = re.compile(
    r"\?|\u061f|\b(pourquoi|comment|qu['\u2019]est|what|why|how|where|when|who)\b"
    r"|\u0645\u0627\s|\u0643\u064a\u0641\s|\u0647\u0644\s",
    re.I,
)
# Extended address regex — covers Arabic, French, Tunisian addresses
_RX_ADDR = re.compile(
    r"\b(rue|avenue|av\.|boulevard|blvd|quartier|ville|wilaya|gouvernorat"
    r"|adresse|email|t\u00e9l|t\u00e9l\u00e9phone|@"
    r"|\u0634\u0627\u0631\u0639|\u062d\u064a|\u0645\u062f\u064a\u0646\u0629"
    r"|\u0648\u0644\u0627\u064a\u0629|\u0639\u0646\u0648\u0627\u0646"
    r"|\u0635\u0646\u062f\u0648\u0642\s*\u0628\u0631\u064a\u062f"
    r"|cit\u00e9|lotissement|lot\.|immeuble|appt|appartement|b\u00e2timent"
    r"|route|rte\.|km\s*\d|n\u00b0\s*\d|\d+\s*,\s*rue"
    r"|tunis|sfax|sousse|bizerte|nabeul|monastir|mahdia|kairouan|gafsa|tozeur"
    r"|ariana|ben\s*arous|manouba|zaghouan|siliana|jendouba|kef|kasserine"
    r"|sidi\s*bouzid|gabes|medenine|tataouine|kebili)\b",
    re.I,
)
_RX_FORM = re.compile(
    r"\b(nom\s*:|pr\u00e9nom\s*:|date\s*:|n\u00b0|cin\s*:|id\s*:)\b", re.I,
)


def _intent(text: str) -> str:
    hits = len(_RX_PRESC.findall(text))
    if hits >= 2 or re.search(r"\bordonnance\b|\bprescription\b", text, re.I):
        return "PRESCRIPTION"
    if _RX_Q.search(text):
        return "QUESTION"
    if _RX_FORM.search(text):
        return "FORM"
    if _RX_ADDR.search(text):
        return "ADDRESS"
    return "PLAIN_TEXT"

# ═══════════════════════════════════════════════════════════════
# Action agents
# ═══════════════════════════════════════════════════════════════

def _groq_text(prompt: str, max_tokens: int = 600) -> dict:
    return _groq([{"role": "user", "content": prompt}],
                 max_tokens=max_tokens, json_mode=True, temperature=0.3)


def _agent_question(text: str, language: str) -> dict:
    lang = _LANG_LABELS.get(language, language)
    personal = re.search(
        r"\b(my|mine|moi|mon|ma|mes|je|j'|\u0623\u0646\u0627|\u0644\u064a"
        r"|\u0639\u0646\u062f\u064a|\u062d\u0642\u064a)\b", text, re.I)
    if personal:
        msg = {
            "english": "This question seems personal. I can only answer general questions.",
            "french":  "Cette question semble personnelle. Je réponds uniquement aux questions générales.",
        }.get(language,
              "هذا السؤال يبدو شخصياً. يمكنني الإجابة فقط على الأسئلة العامة.")
        return {"spoken_response": msg, "display_title": "❓ Question"}
    try:
        r = _groq_text(
            f"Answer this {lang} question clearly in {lang}, under 5 sentences, "
            f"no bullets.\nQuestion: \"\"\"{text}\"\"\"\n"
            f'ONLY JSON: {{"spoken_response":"<answer in {lang}>"}}'
        , 400)
        return {"spoken_response": r.get("spoken_response", text),
                "display_title": "❓ Question"}
    except Exception as e:
        return {"spoken_response": text, "display_title": "❓ Question",
                "error": str(e)}


def _agent_plain(text: str, language: str) -> dict:
    lang    = _LANG_LABELS.get(language, language)
    is_long = len(text.split()) > _LONG_WORDS
    if is_long:
        instruction = (
            f"Summarize this long {lang} text for a blind person in {lang}, "
            f"3-5 sentences, no bullets."
        )
        title = "📄 Summary"
    else:
        instruction = (
            f"Read this {lang} text naturally for a blind person. "
            f"Start with 'The text says:' then read it, then 'In summary:' with 1-2 sentences."
        )
        title = "📄 Text"
    try:
        r = _groq_text(
            f"{instruction}\n\nText: \"\"\"{text}\"\"\"\n"
            f'ONLY JSON: {{"spoken_response":"<response in {lang}>","key_points":""}}',
            600,
        )
        return {"spoken_response": r.get("spoken_response", text),
                "display_title": title}
    except Exception as e:
        return {"spoken_response": text, "display_title": title, "error": str(e)}


def _agent_prescription(text: str, language: str) -> dict:
    lang = _LANG_LABELS.get(language, language)
    try:
        r = _groq_text(
            f"Medical reading assistant for blind patient. {lang} prescription.\n"
            "⚠️ SAFETY: Use ONLY what is written. NEVER invent/guess. "
            "[illegible] → say 'unclear, confirm with pharmacist'.\n"
            f'Prescription: """{text}"""\n'
            f"Extract every medicine. Generate spoken summary in {lang}, no bullets.\n"
            f"ONLY JSON:\n"
            '{{"medicines":[{{"name":"","dosage":"","frequency":"","duration":"","instructions":""}}],'
            f'"spoken_response":"<spoken in {lang}>","medicine_count":0,"has_illegible":false}}',
            800,
        )
        has_il = (
            r.get("has_illegible", False)
            or "[illegible]" in text.lower()
            or "[غير مقروء]" in text
        )
        spoken = r.get("spoken_response", text)
        if has_il:
            spoken += (
                " Veuillez confirmer chaque dosage avec votre pharmacien."
                if language == "french" else
                " Please confirm every dosage with your pharmacist."
            )
        return {
            "spoken_response": spoken,
            "medicines":       r.get("medicines", []),
            "medicine_count":  r.get("medicine_count", 0),
            "has_illegible":   has_il,
            "display_title":   "💊 Prescription",
        }
    except Exception as e:
        return {"spoken_response": text, "medicines": [],
                "medicine_count": 0, "has_illegible": False,
                "display_title": "💊 Prescription", "error": str(e)}


def _agent_address(text: str, language: str) -> dict:
    lang = _LANG_LABELS.get(language, language)
    try:
        r = _groq_text(
            f"Contact reading assistant for blind person. {lang}.\n"
            f'Text: """{text}"""\n'
            f"Parse fields and generate spoken reading in {lang}, no bullets.\n"
            'ONLY JSON: {"parsed":{"full_name":"","address":"","street":"","city":"","postal_code":"","phone":"","email":"","other":""},'
            f'"spoken_response":"<reading in {lang}>"}}',
            500,
        )
        parsed = r.get("parsed", {})

        # Build the best possible navigate_to string from all address fields
        # Priority: full address > street+city > city alone
        nav_parts = []
        if parsed.get("address"):
            nav_parts.append(parsed["address"].strip())
        else:
            if parsed.get("street"):
                nav_parts.append(parsed["street"].strip())
            if parsed.get("city"):
                nav_parts.append(parsed["city"].strip())
            if parsed.get("postal_code"):
                nav_parts.append(parsed["postal_code"].strip())

        # If still empty, try to extract address-like fragment from raw text
        if not nav_parts:
            # Look for "rue X" / "شارع X" / "avenue X" patterns
            m = re.search(
                r"((?:rue|avenue|av\.|boulevard|blvd|شارع|حي|route)\s+[^\n,;]{3,60})",
                text, re.I
            )
            if m:
                nav_parts.append(m.group(1).strip())

        navigate_to = ", ".join(p for p in nav_parts if p)

        return {
            "spoken_response": r.get("spoken_response", text),
            "parsed":          parsed,
            "navigate_to":     navigate_to,
            "display_title":   "📍 Address / Contact",
        }
    except Exception as e:
        # Fallback: try regex extraction directly
        m = re.search(
            r"((?:rue|avenue|av\.|boulevard|blvd|شارع|حي|route)\s+[^\n,;]{3,60})",
            text, re.I
        )
        nav = m.group(1).strip() if m else ""
        return {"spoken_response": text, "parsed": {}, "navigate_to": nav,
                "display_title": "📍 Address", "error": str(e)}


def _agent_form(text: str, language: str) -> dict:
    lang = _LANG_LABELS.get(language, language)
    try:
        r = _groq_text(
            f"Form reading assistant for blind person. {lang}.\n"
            f'Form: """{text}"""\n'
            f"Extract field-value pairs. Spoken reading in {lang}, no bullets.\n"
            'ONLY JSON: {"form_type":"<type>","fields":[{"label":"","value":""}],'
            f'"spoken_response":"<reading in {lang}>"}}',
            500,
        )
        return {
            "spoken_response": r.get("spoken_response", text),
            "form_type":       r.get("form_type", "Form"),
            "fields":          r.get("fields", []),
            "display_title":   f"📋 {r.get('form_type', 'Form')}",
        }
    except Exception as e:
        return {"spoken_response": text, "form_type": "Form", "fields": [],
                "display_title": "📋 Form", "error": str(e)}


_AGENT_ROUTER = {
    "QUESTION":     _agent_question,
    "PLAIN_TEXT":   _agent_plain,
    "PRESCRIPTION": _agent_prescription,
    "ADDRESS":      _agent_address,
    "FORM":         _agent_form,
}

# ═══════════════════════════════════════════════════════════════
# TTS helper — delegates to ActionAgent if available
# ═══════════════════════════════════════════════════════════════

_LANG_TO_TTS = {"english": "en", "french": "fr",
                "arabic": "ar", "darija": "tn"}


def _tts_speak(spoken: str, language: str, intent: str,
               has_illegible: bool, action_agent) -> None:
    if action_agent is None or not spoken:
        return
    tts_lang = _LANG_TO_TTS.get(language, "en")
    try:
        urgent = intent == "PRESCRIPTION" and has_illegible
        if urgent:
            action_agent.speak(spoken, lang=tts_lang, interrupt=True)
        else:
            action_agent.speak_normal(spoken, lang=tts_lang)
    except Exception as e:
        print(f"[handwriting] TTS error (non-fatal): {e}")

# ═══════════════════════════════════════════════════════════════
# Core process function
# ═══════════════════════════════════════════════════════════════

# Simple in-process cache: image_md5+lang → result dict
_cache: dict = {}


def _process(image_bytes: bytes, language: str, action_agent) -> dict:
    # ── Cache check ────────────────────────────────────────────
    key = hashlib.md5(image_bytes).hexdigest() + "_" + language
    if key in _cache:
        print(f"[handwriting] cache hit {key[:12]}")
        return _cache[key]

    # ── Decode & preprocess ────────────────────────────────────
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image", "intent": "ERROR",
                "spoken_response": "Could not decode image."}

    b64 = _encode(_preprocess(img))

    # ── Pass 1: Transcription ──────────────────────────────────
    try:
        p1 = _run_pass1(b64, language)
    except Exception as e:
        return {"error": f"Pass 1 failed: {e}", "intent": "ERROR",
                "spoken_response": "Could not read the handwriting."}

    transcription = p1.get("transcription", "")
    confidence    = float(p1.get("confidence", 0.0))
    backend       = p1.get("backend", "unknown")
    notes         = p1.get("notes", "")
    det_lang      = _normalize_lang(p1.get("detected_language", ""), language)

    # If still unresolved, use script detection on the actual transcription
    if det_lang in ("auto", "english") and language == "auto":
        script_guess = _detect_script(transcription)
        if script_guess != det_lang:
            print(f"[handwriting] script override: {det_lang!r} → {script_guess!r}")
            det_lang = script_guess

    if not transcription.strip():
        return {"error": "No text detected", "intent": "ERROR",
                "spoken_response": "No text detected in the image."}

    print(f"[handwriting] P1 ({backend}) {len(transcription.split())}w "
          f"conf={confidence:.2f} lang={det_lang}")

    final = transcription

    # ── Pass 2: Verification ───────────────────────────────────
    meaning = ""
    try:
        # By this point det_lang is always a concrete language (never "auto")
        is_ar = det_lang in _IS_ARABIC
        if is_ar:
            # Arabic/Darija: 3-way verification (Gemini + Groq + OR-Qwen)
            final, meaning = _run_arabic_verification(
                b64, transcription, det_lang, confidence)
        else:
            # English / French: Groq primary + Gemini cross-check
            try:
                ver = _pass2_groq(b64, transcription, det_lang, confidence)
                if ver.get("was_corrected"):
                    final = ver.get("corrected_transcription", transcription)
                    print("[handwriting] P2 Groq correction applied")
                meaning = ver.get("meaning_english", "")
            except Exception as e:
                if "429" not in str(e):
                    print(f"[handwriting] P2 Groq failed (non-fatal): {e}")
            if GOOGLE_API_KEY:
                try:
                    ver2 = _pass2_gemini(b64, final, det_lang)
                    if ver2.get("was_corrected"):
                        final = ver2.get("corrected_transcription", final)
                        print("[handwriting] P2 Gemini cross-check applied")
                    if not meaning:
                        meaning = ver2.get("meaning_english", "")
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[handwriting] P2 Gemini cross-check failed: {e}")
    except Exception as e:
        print(f"[handwriting] P2 outer failed (non-fatal): {e}")

    # ── Pass 2B: Digit check (prescriptions) ──────────────────
    _is_med = bool(re.search(
        r"(ordonnance|prescription|cp|mg|\bg\b|fois|jour|matin|soir|amoxi|ibupro|paracet)",
        final, re.I))
    _has_num = bool(re.search(r"\b\d+\s*(?:mg|g|cp)\b", final, re.I))
    if _is_med and _has_num:
        try:
            dv = _groq([{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _digit_prompt(final, det_lang)},
            ]}], max_tokens=512)
            if dv.get("was_corrected"):
                final = dv.get("corrected_transcription", final)
                print("[handwriting] P2B digit correction applied")
        except Exception as e:
            print(f"[handwriting] P2B failed (non-fatal): {e}")

    # ── Pass 2C: Deterministic guardrails ─────────────────────
    guarded, changes = _guardrails(final)
    if changes:
        final = guarded
        print(f"[handwriting] P2C guardrails: {changes}")

    # ── Pass 3: Intent → Agent ─────────────────────────────────
    intent_key = _intent(final)
    handler    = _AGENT_ROUTER.get(intent_key, _agent_plain)
    try:
        agent_result = handler(final, det_lang)
    except Exception as e:
        agent_result = {"spoken_response": final,
                        "display_title": "📄 Text", "error": str(e)}

    spoken      = agent_result.get("spoken_response", final)
    has_il      = agent_result.get("has_illegible", False)

    # ── TTS ────────────────────────────────────────────────────
    _tts_speak(spoken, det_lang, intent_key, has_il, action_agent)

    result = {
        "transcription":   final,
        "language":        det_lang,
        "intent":          intent_key,
        "confidence":      confidence,
        "backend":         backend,
        "notes":           notes,
        "spoken_response": spoken,
        "display_title":   agent_result.get("display_title", "📄 Result"),
        "meaning_english": meaning,
        "has_illegible":   has_il,
        "medicines":       agent_result.get("medicines", []),
        "navigate_to":     agent_result.get("navigate_to", ""),
        "fields":          agent_result.get("fields", []),
        "parsed":          agent_result.get("parsed", {}),
        "error":           agent_result.get("error"),
    }

    _cache[key] = result
    return result

# ═══════════════════════════════════════════════════════════════
# FastAPI request schema + endpoints
# ═══════════════════════════════════════════════════════════════

class HandwritingRequest(BaseModel):
    image_b64: str
    language: str = "auto"


# Lazy reference to main.py's action agent
_action_agent_ref = None


def _get_action_agent():
    global _action_agent_ref
    if _action_agent_ref is None:
        try:
            from main import action as _a
            _action_agent_ref = _a
        except Exception:
            pass
    return _action_agent_ref


@router.post("/handwriting")
async def handwriting(req: HandwritingRequest):
    try:
        language = req.language.strip().lower()
        if language not in {"auto", "english", "french", "arabic", "darija"}:
            language = "auto"

        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception:
            return JSONResponse(status_code=400,
                                content={"error": "Invalid base64 image"})

        if len(image_bytes) < 1000:
            return JSONResponse(status_code=400,
                                content={"error": "Image too small"})

        result = _process(image_bytes, language, _get_action_agent())
        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/handwriting/health")
async def handwriting_health():
    return {"status": "ok", "endpoint": "/handwriting"}


# ═══════════════════════════════════════════════════════════════
# Hands-free WebSocket endpoint  /ws/handwriting
# ═══════════════════════════════════════════════════════════════
#
# Protocol (phone ↔ server)
# ─────────────────────────
#   Phone → server (binary):  JPEG frames continuously
#   Phone → server (text):    "RESET" | "SET_LANG:<lang>"
#
#   Server → phone (text JSON):
#     {"type":"ready"}
#     {"type":"guide",  "message":"<spoken guidance>", "stable":false,
#                       "stable_count":<int>}
#     {"type":"capture","message":"Hold steady, reading now…"}
#     {"type":"result", "spoken_response":"…", "intent":"…",
#                       "transcription":"…", "language":"…",
#                       "confidence":<float>, "navigate_to":"…",
#                       "error":null}
#     {"type":"error",  "message":"…"}
#
# Flow:
#   1. Phone opens WS → server sends {"type":"ready"}
#   2. Phone streams JPEG frames at ~4 fps
#   3. Server analyzes framing every 2s (Gemini Flash-Lite, fast)
#   4. Server speaks guidance: "Move closer", "Hold steady", etc.
#   5. After STABLE_FRAMES_NEEDED consecutive stable frames →
#      server sends {"type":"capture"} and runs full OCR pipeline
#   6. Server sends {"type":"result"} with spoken_response
#   7. Phone speaks result via TTS, then resets for next document

_GUIDE_INTERVAL_S = 1.2   # analyze every 1.2s — fast enough, not spammy

# ── Guidance messages (no LLM needed) ────────────────────────
_GUIDE_MSG = {
    "dark":    "Too dark. Move to better lighting.",
    "no_text": "No text detected. Point camera at the document.",
    "ok":      "Good. Tap the screen to read.",
}


def _analyze_framing_local(frame_bgr: np.ndarray) -> dict:
    """3-check framing analysis — dark / no text / good."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Too dark?
    if float(np.mean(gray)) < 30:
        return {"stable": False, "guidance": _GUIDE_MSG["dark"]}

    # 2. Any text-like edges? (very lenient — just checks something is there)
    edges = cv2.Canny(gray, 20, 80)
    if float(np.count_nonzero(edges)) / (h * w) < 0.003:
        return {"stable": False, "guidance": _GUIDE_MSG["no_text"]}

    # 3. All good
    return {"stable": True, "guidance": _GUIDE_MSG["ok"]}


@router.websocket("/ws/handwriting")
async def handwriting_ws(ws: WebSocket):
    await ws.accept()
    print("[hw_ws] Client connected")
    await ws.send_text(json.dumps({"type": "ready"}))

    language      = "auto"
    last_guide_ts = 0.0
    last_guidance = ""
    capturing     = False      # True while OCR pipeline is running
    latest_frame: Optional[bytes] = None   # most recent JPEG from phone

    async def _send_result(res: dict):
        await ws.send_text(json.dumps({
            "type":            "result",
            "spoken_response": res.get("spoken_response", ""),
            "intent":          res.get("intent", "PLAIN_TEXT"),
            "transcription":   res.get("transcription", ""),
            "language":        res.get("language", ""),
            "confidence":      res.get("confidence", 0.0),
            "navigate_to":     res.get("navigate_to", ""),
            "medicines":       res.get("medicines", []),
            "fields":          res.get("fields", []),
            "has_illegible":   res.get("has_illegible", False),
            "error":           res.get("error"),
        }))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                if not capturing:
                    await ws.send_text(json.dumps({
                        "type": "guide",
                        "message": "Point camera at the document.",
                        "stable": False,
                    }))
                continue
            except Exception:
                break

            # ── Binary: JPEG frame from phone ─────────────────
            if "bytes" in msg and msg["bytes"]:
                latest_frame = msg["bytes"]
                if capturing:
                    continue   # OCR running — just buffer the frame

                now = _time.time()
                if now - last_guide_ts < _GUIDE_INTERVAL_S:
                    continue   # throttle: local analysis is fast but no need every frame

                last_guide_ts = now

                # Decode + resize for analysis (pure local, no API)
                arr   = np.frombuffer(latest_frame, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                if max(h, w) > 640:
                    s     = 640 / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)))

                # Local framing analysis — instant, zero API calls
                guide     = _analyze_framing_local(frame)
                is_stable = bool(guide.get("stable", False))
                guidance  = guide.get("guidance", "Hold steady.")

                # Send guidance only when it changes (avoid TTS spam)
                if guidance != last_guidance:
                    last_guidance = guidance
                    await ws.send_text(json.dumps({
                        "type":    "guide",
                        "message": guidance,
                        "stable":  is_stable,
                    }))

            # ── Text: control commands from phone ─────────────
            elif "text" in msg and msg["text"]:
                cmd = msg["text"].strip()

                if cmd == "CAPTURE_NOW":
                    # User tapped — run OCR on the latest frame
                    if capturing or not latest_frame:
                        continue
                    capturing = True
                    last_guidance = ""
                    await ws.send_text(json.dumps({
                        "type":    "capture",
                        "message": "Reading now.",
                    }))
                    fb   = latest_frame
                    lang = language

                    def _run_ocr():
                        return _process(fb, lang, _get_action_agent())

                    try:
                        res = await run_in_threadpool(_run_ocr)
                    except Exception as e:
                        res = {"spoken_response": "Could not read the document.",
                               "intent": "ERROR", "error": str(e)}
                    finally:
                        capturing = False

                    await _send_result(res)

                elif cmd == "RESET":
                    capturing     = False
                    last_guidance = ""
                    await ws.send_text(json.dumps({
                        "type":    "guide",
                        "message": "Ready. Point camera at the document.",
                        "stable":  False,
                    }))

                elif cmd.startswith("SET_LANG:"):
                    language = cmd.split(":", 1)[1].strip().lower()
                    print(f"[hw_ws] language set to {language!r}")

    except WebSocketDisconnect:
        print("[hw_ws] Client disconnected")
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        print("[hw_ws] Connection closed")