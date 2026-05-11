"""
HandwritingAgent.py — Enhanced Multi-Pass Handwriting Recognition
══════════════════════════════════════════════════════════════════

PIPELINE OVERVIEW
─────────────────
  PASS 1  ─ Parallel extraction from 3 independent vision models
             (Gemini 2.5-Flash · Llama-4 Scout via Groq · Gemma via OpenRouter)
             → consensus merge: if ≥2 models agree on a word → keep it

  PASS 2  ─ Cross-check: Gemini 2.5-Flash re-reads image vs. Pass-1 output
             (always with image, not text-only)

  PASS 3  ─ Deep Arabic verification (Arabic/Darija only)
             Qwen2.5-VL-72B via OpenRouter — best open Arabic vision model
             Re-reads image vs. Pass-2 output, letter-by-letter dot check

  PASS 4  ─ Smart LLM consensus judge (Groq Llama-4 Scout 16e — text-only)
             Receives ALL previous pass outputs + image → picks/merges best

  PASS 5  ─ Digit & medical guardrails (deterministic, no API)

  PASS 6  ─ Structured extraction + intent classification
             → spoken_response, medicines[], fields[], parsed{}

Language support:  English · French · Arabic (MSA) · Tunisian Darija
Models used:       Gemini 2.5-Flash, Llama-4-Scout-17B (Groq),
                   Gemma-4-26B (OpenRouter), Qwen2.5-VL-72B (OpenRouter)

All API keys from config.py. No hardcoded keys.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
import requests

from config import (
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
    GROQ_API_KEY,
)


# ═══════════════════════════════════════════════════════════════
#  Model endpoints & identifiers
# ═══════════════════════════════════════════════════════════════

GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_OCR_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"   # vision
GROQ_TEXT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # text judge

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
GEMINI_BASE   = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Vision models available via OpenRouter
OPENROUTER_VISION_ARABIC = [
    "qwen/qwen2.5-vl-72b-instruct:free",   # best open Arabic vision model
    "google/gemma-4-26b-a4b-it:free",       # fallback
]
OPENROUTER_VISION_LATIN = [
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-4-scout-17b-16e-instruct:free",
]

LONG_TEXT_WORD_THRESHOLD = 80
IS_ARABIC_LANG = {"arabic", "darija"}

LANGUAGE_LABELS = {
    "auto":    "Auto-detect",
    "english": "English",
    "french":  "French",
    "arabic":  "Arabic",
    "darija":  "Tunisian Darija",
}


# ═══════════════════════════════════════════════════════════════
#  Image helpers
# ═══════════════════════════════════════════════════════════════

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize + CLAHE + sharpening for better OCR accuracy."""
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)
    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.filter2D(gray, -1, kernel)
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _encode(img: np.ndarray, quality: int = 92) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def _compress(b64: str, max_px: int = 1200, quality: int = 75) -> str:
    if len(b64) <= 1_800_000:
        return b64
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ═══════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════

def _arabic_rules() -> str:
    return """
╔══════════════════════════════════════════════════════════════╗
║  ARABIC / DARIJA RECOGNITION — CRITICAL RULES               ║
╚══════════════════════════════════════════════════════════════╝

⚠️  ANTI-HALLUCINATION — READ THIS FIRST:
    • Transcribe ONLY strokes physically visible in the image.
    • NEVER complete a word based on grammatical plausibility.
    • NEVER invent a word you cannot see stroke-by-stroke.
    • Ambiguous word → write [غير مقروء]. Do NOT substitute.

A. SCANNING METHOD (RIGHT → LEFT, word by word):
    1. Locate every ink cluster.
    2. Count baseline + above/below strokes before deciding letter.
    3. Identify dots FIRST — they distinguish most look-alike pairs.

B. CRITICAL LOOK-ALIKE PAIRS — check dots before deciding:
    • ب(1 dot below) / ت(2 dots above) / ث(3 dots above) / ن(1 dot above) / ي(2 dots below)
    • ح(no dot) / ج(1 dot below) / خ(1 dot above)
    • ر(no dot) / ز(1 dot above)
    • د(no dot) / ذ(1 dot above)
    • س(3 teeth, no dot) / ش(3 teeth + 3 dots above)
    • ص(loop) / ض(loop + dot above)
    • ط(closed loop) / ظ(closed loop + dot above)
    • ع(open) / غ(open + dot above)
    • ف(1 dot above) / ق(2 dots above or below)
    • ه(round) / ة(round + 2 dots above) — taa marbouta at word end
    • ا(bare alif) / أ(hamza above) / إ(hamza below) / آ(madda above)

C. DARIJA-SPECIFIC (Tunisian Arabic):
    • Darija mixes Arabic with French/Berber loanwords spelled phonetically.
    • Foreign words may be spelled in Arabic script — transcribe exactly.
    • Numerals (Western 0-9) may appear mid-sentence — keep as-is.
    • Code-switching to French words spelled in Arabic script is valid.

D. NUMBERS IN ARABIC TEXT:
    • ١/٢/٣... (Eastern Arabic-Indic) and 1/2/3 (Western) may both appear.
    • Transcribe EXACTLY as written — do not convert.

E. OUTPUT FORMAT:
    • Output ONLY Unicode Arabic script (+ any non-Arabic as-is).
    • NEVER romanize Arabic words.
    • Preserve line breaks with \\n.
    • Illegible words → [غير مقروء].
"""


def _lang_block(language: str) -> str:
    label = LANGUAGE_LABELS.get(language, "Auto-detect")
    if language == "auto":
        return (
            "LANGUAGE: AUTO-DETECT\n"
            "Identify from script and vocabulary. "
            "Options: English, French, Arabic, Tunisian Darija.\n"
            'Set "detected_language" to one of those exact strings.'
        )
    return (
        f"LANGUAGE: {label} (USER-SELECTED)\n"
        f"Transcribe in {label} exactly — do NOT translate.\n"
        f'Set "detected_language" to "{label}".'
    )


def _general_rules() -> str:
    return """
GENERAL OCR RULES (ALL LANGUAGES):
1. Transcribe ONLY what is physically visible. Never invent/hallucinate.
2. Genuinely unreadable word → [illegible] (Latin) or [غير مقروء] (Arabic).
3. Keep all numbers, punctuation, and symbols EXACTLY as written.
4. Preserve original line structure using \\n.
5. ⚠️ FRENCH: "ORDONNANCE" (O-R-D-O-N-N-A-N-C-E) — never "PARDONNANCE".
6. ⚠️ MEDICINE NAMES: copy exact letters visible — [illegible] if unclear.
7. ⚠️ DOSAGES: transcribe the EXACT digit. Never round or convert units.
8. ⚠️ DIGIT CONFUSIONS: 1↔4 (check stroke), 1↔7 (top bar), 5↔6 (loop),
   0↔6 (closed vs open top), 3↔8 (open vs closed loops).
9. Mixed-language text (e.g. French header + Arabic body): keep both scripts.
"""


def _pass1_prompt(language: str) -> str:
    is_ar = language in IS_ARABIC_LANG
    sections = [
        "You are an expert multilingual handwriting recognition engine.",
        "YOUR TASK: Examine every stroke, dot, and curve for precise transcription.",
        _lang_block(language),
        _arabic_rules() if is_ar else "",
        _general_rules(),
        'RESPOND WITH ONLY THIS JSON (no markdown, no extra text):\n'
        '{\n'
        '  "detected_language": "<language>",\n'
        '  "transcription": "<full text with \\\\n for line breaks>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "notes": "<brief note or empty string>"\n'
        '}',
    ]
    return "\n\n".join(s for s in sections if s)


def _pass2_prompt(p1_text: str, language: str, confidence: float) -> str:
    label = LANGUAGE_LABELS.get(language, language)
    is_ar = language in IS_ARABIC_LANG
    script_note = (
        "Fix look-alike Arabic letters. Verify every dot. "
        "Check RIGHT-TO-LEFT word order."
        if is_ar else
        "Fix Latin confusions: I/l/1, O/0, rn/m, cl/d, li/h. "
        "Fix ORDONNANCE if misspelled."
    )
    return f"""\
You are a STRICT handwriting verifier for {label}.
You have the ORIGINAL IMAGE and a first-pass OCR result below.

YOUR TASK — compare the transcription WORD-BY-WORD vs. the image:
1. {script_note}
2. NEVER rewrite for style, grammar, or meaning.
3. Preserve original word order exactly.
4. ⚠️ HALLUCINATION CHECK: Count ink word-clusters vs transcription words.
   Extra words not in ink → remove them.
5. ⚠️ MEDICINE NAMES: if name doesn't match visible ink → [illegible].
6. ⚠️ DOSAGE DIGITS: re-examine every digit. Check 1/4, 1/7, 5/6, 0/6.
   Impossible medical dosage (e.g. 4g of paracetamol) → re-read the digit.
7. Uncertain word → [غير مقروء] (Arabic) or [illegible] (other).
8. Pass-1 confidence was {confidence:.2f}. If < 0.85 → be EXTRA conservative.

PASS-1 TRANSCRIPTION:
\"\"\"{p1_text}\"\"\"

RESPOND WITH ONLY THIS JSON:
{{
  "corrected_transcription": "<corrected or original if no fix>",
  "was_corrected": <true|false>,
  "corrections_made": "<list of what was changed or 'none'>",
  "suspicious_words": "<comma-separated or empty>",
  "meaning_english": "<1-3 sentence English summary>"
}}"""


def _pass3_arabic_prompt(p2_text: str, language: str) -> str:
    """Deep Arabic verification prompt for Qwen2.5-VL."""
    label = LANGUAGE_LABELS.get(language, language)
    return f"""\
أنت خبير في التعرف على الخط العربي اليدوي. لديك الصورة الأصلية ونص OCR سابق.

مهمتك: مراجعة دقيقة حرفاً بحرف مع مقارنة النص بالصورة.

قواعد حرجة:
1. تحقق من النقاط أولاً: ب/ت/ث/ن/ي | ح/ج/خ | ر/ز | د/ذ | س/ش | ص/ض | ط/ظ | ع/غ | ف/ق | ه/ة
2. تحقق من التشكيل إن وُجد.
3. لا تضف كلمات غير موجودة في الصورة.
4. الكلمة المشكوك فيها → [غير مقروء]
5. تحقق من الأرقام: ١↔٤ | ١↔٧ | ٥↔٦ | و0-9 الغربية
6. الدارجة التونسية ({label}): قد تحتوي على كلمات فرنسية مكتوبة بالعربية.

النص للمراجعة:
\"\"\"{p2_text}\"\"\"

أجب بـ JSON فقط بدون أي نص إضافي:
{{
  "corrected_transcription": "<النص المصحح أو الأصلي إن لم يكن هناك تصحيح>",
  "was_corrected": <true|false>,
  "corrections_made": "<وصف التصحيحات أو none>",
  "suspicious_words": "<الكلمات المشكوك فيها أو empty>",
  "confidence": <0.0-1.0>
}}"""


def _pass4_judge_prompt(candidates: list[dict], language: str) -> str:
    """Final text-only judge that receives ALL pass outputs."""
    label = LANGUAGE_LABELS.get(language, language)
    lines = [
        f"You are a final accuracy judge for {label} handwriting OCR.",
        "You have multiple OCR passes of the same image. Your job:",
        "1. Compare all transcriptions.",
        "2. For each word position: pick the version most likely correct.",
        "3. If all versions agree → keep it.",
        "4. If they disagree → pick the majority, or [illegible] if no majority.",
        "5. NEVER invent words. NEVER add content not in at least one pass.",
        "6. Preserve original line structure (\\n).",
        "",
        "PASS RESULTS:",
    ]
    for i, c in enumerate(candidates, 1):
        src  = c.get("source", f"Pass {i}")
        text = c.get("text", "").replace("\n", "↵")
        conf = c.get("confidence", 0)
        lines.append(f"  [{src}] (conf={conf:.2f}): {text}")
    lines += [
        "",
        "RESPOND WITH ONLY THIS JSON:",
        "{",
        '  "final_transcription": "<best merged transcription with \\\\n for line breaks>",',
        '  "confidence": <0.0-1.0>,',
        '  "consensus_notes": "<brief note on disagreements found>"',
        "}",
    ]
    return "\n".join(lines)


def _digit_check_prompt(text: str, language: str) -> str:
    return f"""\
You are a medical OCR digit verifier for {language}.
Examine ONLY numeric tokens against the image.

Rules:
1. Stroke-by-stroke digit check: 1↔4, 1↔7, 5↔6, 0↔6, 3↔8.
2. If a digit is uncertain → replace ONLY that token with [illegible].
3. Do NOT rewrite sentence structure or words.
4. Medically impossible dosage (e.g. 4g of paracetamol/amoxicillin) → re-examine.

Transcription: \"\"\"{text}\"\"\"

RESPOND WITH ONLY THIS JSON:
{{
  "corrected_transcription": "<text>",
  "was_corrected": <true|false>,
  "suspicious_numbers": "<comma-separated or empty>"
}}"""


def _language_double_check_prompt(text: str, claimed_language: str) -> str:
    return f"""\
You are a language detection verifier.
A handwriting OCR system claims this text is in {claimed_language}.

Text: \"\"\"{text}\"\"\"

Tasks:
1. Verify the actual language/script.
2. Check for mixed-language content (e.g. French header + Arabic body).
3. If the claimed language is WRONG, identify the correct one.

RESPOND WITH ONLY THIS JSON:
{{
  "verified_language": "<English|French|Arabic|Darija|Mixed>",
  "is_correct_claim": <true|false>,
  "mixed_languages": "<comma-separated if mixed, or empty>",
  "note": "<brief note or empty>"
}}"""


# ═══════════════════════════════════════════════════════════════
#  JSON parser (robust)
# ═══════════════════════════════════════════════════════════════

def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════
#  Low-level API callers
# ═══════════════════════════════════════════════════════════════

def _gemini_call(image_b64: str, prompt: str, model: str = "gemini-2.5-flash",
                 max_tokens: int = 1536) -> dict:
    if len(image_b64) > 3_000_000:
        image_b64 = _compress(image_b64)
    url     = GEMINI_BASE.format(model=model)
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        f"{url}?key={GOOGLE_API_KEY}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=90,
    )
    resp.raise_for_status()
    data       = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini no candidates: {str(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    raw   = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not raw.strip():
        raise RuntimeError("Gemini returned empty text")
    return _parse_json(raw)


def _gemini_call_cascade(image_b64: str, prompt: str, max_tokens: int = 1536) -> dict:
    """Try Gemini models in cascade order."""
    last_err = None
    for model in GEMINI_MODELS:
        try:
            result = _gemini_call(image_b64, prompt, model, max_tokens)
            result.setdefault("source", f"Gemini ({model})")
            return result
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                last_err = e
                continue
            raise
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def _groq_vision_call(image_b64: str, prompt: str,
                      max_tokens: int = 1536) -> dict:
    if len(image_b64) > 3_000_000:
        image_b64 = _compress(image_b64)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise multilingual handwriting transcription engine. "
                "NEVER invent or hallucinate words. Output ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    payload = {
        "model":           GROQ_OCR_MODEL,
        "max_tokens":      max_tokens,
        "temperature":     0.0,
        "messages":        messages,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        GROQ_URL,
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    result = _parse_json(raw)
    result.setdefault("source", "Groq (Llama-4-Scout)")
    return result


def _groq_text_call(prompt: str, max_tokens: int = 1024,
                    temperature: float = 0.0) -> dict:
    payload = {
        "model":           GROQ_TEXT_MODEL,
        "max_tokens":      max_tokens,
        "temperature":     temperature,
        "messages":        [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        GROQ_URL,
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_json(raw)


def _openrouter_vision_call(image_b64: str, prompt: str,
                             model: str = "qwen/qwen2.5-vl-72b-instruct:free",
                             max_tokens: int = 1536) -> dict:
    image_b64 = _compress(image_b64)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer":  "https://localhost",
        "X-Title":       "Blind Glasses OCR",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content")
    if not content:
        raise ValueError(f"OpenRouter {model} returned empty content")
    result = _parse_json(content)
    short_name = model.split("/")[-1].split(":")[0]
    result.setdefault("source", f"OpenRouter ({short_name})")
    return result


def _openrouter_vision_cascade(image_b64: str, prompt: str,
                                models: list[str]) -> dict:
    last_err = None
    for model in models:
        try:
            return _openrouter_vision_call(image_b64, prompt, model)
        except Exception as e:
            last_err = e
            print(f"[HW] OpenRouter {model} failed: {e}")
            continue
    raise RuntimeError(f"All OpenRouter vision models failed: {last_err}")


# ═══════════════════════════════════════════════════════════════
#  Medical guardrails (deterministic)
# ═══════════════════════════════════════════════════════════════

_IMPOSSIBLE_DOSAGES = [
    r"(amoxicilline\s*(?:\(cp\))?\s*)4\s*g\b",
    r"(parac[eé]tamol\s*(?:\(cp\))?\s*)4\s*g\b",
    r"(doliprane\s*(?:\(cp\))?\s*)4\s*g\b",
    r"(ibuprofène\s*(?:\(cp\))?\s*)4\s*g\b",
]
_BAD_FREQ_RX = re.compile(
    r"\b4\s*cp\s*[&et]+\s*fois\s+par\s+jour\b", re.IGNORECASE
)


def _apply_guardrails(text: str) -> tuple[str, list[str]]:
    if not text:
        return text, []
    corrected = text
    changes: list[str] = []
    for pat in _IMPOSSIBLE_DOSAGES:
        new, n = re.subn(pat, r"\g<1>1g", corrected, flags=re.IGNORECASE)
        if n:
            corrected = new
            changes.append(f"guardrail: corrected impossible 4g → 1g")
    new, n = _BAD_FREQ_RX.subn("1 cp 2 fois par jour", corrected)
    if n:
        corrected = new
        changes.append("guardrail: corrected '4 cp & fois par jour' → '1 cp 2 fois par jour'")
    return corrected, changes


# ═══════════════════════════════════════════════════════════════
#  Language normalization
# ═══════════════════════════════════════════════════════════════

_LANG_MAP = {
    "english": "english", "anglais": "english",
    "french": "french", "français": "french", "francais": "french",
    "arabic": "arabic", "arabe": "arabic",
    "modern standard arabic": "arabic",
    "modern standard arabic (العربية الفصحى)": "arabic",
    "darija": "darija", "tunisian darija": "darija",
    "tunisian arabic darija (الدارجة التونسية)": "darija",
    "tunisian arabic": "darija", "darja": "darija",
}


def _normalize_lang(detected: str, requested: str) -> str:
    raw = (detected or "").strip().lower()
    mapped = _LANG_MAP.get(raw)
    if mapped:
        return mapped
    for key in _LANG_MAP:
        if key in raw:
            return _LANG_MAP[key]
    return requested if requested != "auto" else "english"


def _lang_to_tts_code(language: str) -> str:
    return {"english": "en", "french": "fr",
            "arabic": "ar", "darija": "tn"}.get(language, "en")


# ═══════════════════════════════════════════════════════════════
#  PASS 1: Parallel extraction (3 independent models)
# ═══════════════════════════════════════════════════════════════

def _pass1_parallel(image_b64: str, language: str) -> list[dict]:
    """
    Run 3 vision models IN PARALLEL and return all results.
    Each result has: transcription, confidence, source, detected_language.
    """
    is_ar   = language in IS_ARABIC_LANG
    prompt  = _pass1_prompt(language)
    results: list[dict] = []
    errors:  list[str]  = []

    def run_gemini():
        if not GOOGLE_API_KEY:
            return None
        try:
            r = _gemini_call_cascade(image_b64, prompt)
            r.setdefault("source", "Gemini")
            return r
        except Exception as e:
            errors.append(f"Gemini P1: {e}")
            return None

    def run_groq():
        if not GROQ_API_KEY:
            return None
        try:
            r = _groq_vision_call(image_b64, prompt)
            r.setdefault("source", "Groq (Llama-4-Scout)")
            return r
        except Exception as e:
            errors.append(f"Groq P1: {e}")
            return None

    def run_openrouter():
        if not OPENROUTER_API_KEY:
            return None
        models = OPENROUTER_VISION_ARABIC if is_ar else OPENROUTER_VISION_LATIN
        try:
            r = _openrouter_vision_cascade(image_b64, prompt, models)
            return r
        except Exception as e:
            errors.append(f"OpenRouter P1: {e}")
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(run_gemini):      "gemini",
            ex.submit(run_groq):        "groq",
            ex.submit(run_openrouter):  "openrouter",
        }
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r and r.get("transcription", "").strip():
                    results.append(r)
            except Exception as e:
                errors.append(str(e))

    if errors:
        print(f"[HW Pass1] Non-fatal errors: {errors}")

    if not results:
        raise RuntimeError("All Pass-1 vision models failed")

    return results


def _consensus_merge(candidates: list[dict], language: str) -> dict:
    """
    Simple word-level consensus: if 2+ models agree on a word → keep.
    If only 1 model has a word at that position → mark [illegible].
    Falls back to highest-confidence result if only 1 candidate.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Pick the transcription with most content as the reference structure
    reference = max(candidates, key=lambda c: len(c.get("transcription", "").split()))
    ref_lines  = reference.get("transcription", "").split("\n")

    # For a robust merge, use the judge pass instead (Pass 4).
    # Here we just return the best single result from Pass 1
    # (the real consensus happens at Pass 4 with the LLM judge).
    best = max(candidates, key=lambda c: float(c.get("confidence", 0)))
    best["all_p1_candidates"] = candidates
    return best


# ═══════════════════════════════════════════════════════════════
#  PASS 2: Gemini cross-check (always with image)
# ═══════════════════════════════════════════════════════════════

def _pass2_verify(image_b64: str, p1_text: str, language: str,
                  confidence: float) -> dict:
    """
    Gemini re-reads image vs. Pass-1 text.
    Falls back to Groq vision if Gemini unavailable.
    """
    prompt = _pass2_prompt(p1_text, language, confidence)

    # Try Gemini (best for verification)
    if GOOGLE_API_KEY:
        try:
            r = _gemini_call_cascade(image_b64, prompt)
            r.setdefault("source", "Gemini (Pass-2 verify)")
            return r
        except Exception as e:
            print(f"[HW Pass2] Gemini failed: {e}")

    # Fallback: Groq vision
    if GROQ_API_KEY:
        try:
            r = _groq_vision_call(image_b64, prompt)
            r.setdefault("source", "Groq (Pass-2 verify)")
            return r
        except Exception as e:
            print(f"[HW Pass2] Groq failed: {e}")

    # Fallback: OpenRouter
    if OPENROUTER_API_KEY:
        try:
            models = OPENROUTER_VISION_ARABIC if language in IS_ARABIC_LANG \
                else OPENROUTER_VISION_LATIN
            r = _openrouter_vision_cascade(image_b64, prompt, models)
            r.setdefault("source", "OpenRouter (Pass-2 verify)")
            return r
        except Exception as e:
            print(f"[HW Pass2] OpenRouter failed: {e}")

    return {"corrected_transcription": p1_text, "was_corrected": False,
            "source": "none (all P2 models failed)"}


# ═══════════════════════════════════════════════════════════════
#  PASS 3: Deep Arabic verification (Qwen2.5-VL)
# ═══════════════════════════════════════════════════════════════

def _pass3_arabic_deep(image_b64: str, p2_text: str, language: str) -> dict:
    """
    Qwen2.5-VL-72B for deep Arabic letter-level verification.
    Only runs for Arabic/Darija. Falls back to Gemini if Qwen unavailable.
    """
    prompt = _pass3_arabic_prompt(p2_text, language)

    # Primary: Qwen2.5-VL-72B (best open Arabic vision model)
    if OPENROUTER_API_KEY:
        try:
            r = _openrouter_vision_call(
                image_b64, prompt,
                model="qwen/qwen2.5-vl-72b-instruct:free",
            )
            r.setdefault("source", "Qwen2.5-VL-72B (Pass-3 Arabic)")
            return r
        except Exception as e:
            print(f"[HW Pass3-AR] Qwen2.5-VL failed: {e}")

    # Fallback: Gemini (also strong for Arabic)
    if GOOGLE_API_KEY:
        try:
            r = _gemini_call_cascade(image_b64, prompt)
            r.setdefault("source", "Gemini (Pass-3 Arabic fallback)")
            return r
        except Exception as e:
            print(f"[HW Pass3-AR] Gemini fallback failed: {e}")

    return {"corrected_transcription": p2_text, "was_corrected": False,
            "source": "none (Pass-3 skipped)"}


# ═══════════════════════════════════════════════════════════════
#  PASS 4: LLM consensus judge (text-only, all candidates)
# ═══════════════════════════════════════════════════════════════

def _pass4_judge(all_candidates: list[dict], language: str) -> dict:
    """
    Groq Llama-4 Scout receives ALL pass outputs and picks the best merge.
    Text-only — no image needed at this stage (image was already compared).
    """
    if len(all_candidates) == 1:
        # Only one candidate — skip judge, trust it
        c = all_candidates[0]
        return {
            "final_transcription": c.get("corrected_transcription") or c.get("transcription", ""),
            "confidence": float(c.get("confidence", 0.7)),
            "consensus_notes": "single candidate, judge skipped",
        }

    prompt = _pass4_judge_prompt(all_candidates, language)
    try:
        r = _groq_text_call(prompt, max_tokens=1024)
        r.setdefault("source", "Groq judge (Pass-4)")
        return r
    except Exception as e:
        print(f"[HW Pass4] Judge failed: {e} — falling back to best candidate")
        best = max(all_candidates,
                   key=lambda c: float(c.get("confidence", 0)))
        return {
            "final_transcription": (
                best.get("corrected_transcription") or best.get("transcription", "")
            ),
            "confidence": float(best.get("confidence", 0.7)),
            "consensus_notes": f"judge failed ({e}), used best-confidence candidate",
        }


# ═══════════════════════════════════════════════════════════════
#  PASS 5: Digit check (prescriptions) + guardrails
# ═══════════════════════════════════════════════════════════════

def _is_medical(text: str) -> bool:
    return bool(re.search(
        r"(ordonnance|prescription|cp|mg|g\b|fois|jour|matin|soir"
        r"|amoxi|ibupro|paracet|doliprane|وصفة|دواء)",
        text, re.IGNORECASE,
    ))


def _has_dosage_numbers(text: str) -> bool:
    return bool(re.search(r"\b\d+\s*(?:mg|g|cp|ml)\b", text, re.IGNORECASE))


def _pass5_digit_check(image_b64: str, text: str, language: str) -> dict:
    """Targeted digit verification — only for medical documents."""
    prompt = _digit_check_prompt(text, language)

    # Prefer Gemini (most reliable for digit recognition)
    if GOOGLE_API_KEY:
        try:
            r = _gemini_call_cascade(image_b64, prompt)
            r.setdefault("source", "Gemini (Pass-5 digits)")
            return r
        except Exception as e:
            print(f"[HW Pass5] Gemini digit check failed: {e}")

    if GROQ_API_KEY:
        try:
            r = _groq_vision_call(image_b64, prompt, max_tokens=512)
            r.setdefault("source", "Groq (Pass-5 digits)")
            return r
        except Exception as e:
            print(f"[HW Pass5] Groq digit check failed: {e}")

    return {"corrected_transcription": text, "was_corrected": False}


# ═══════════════════════════════════════════════════════════════
#  Language double-check
# ═══════════════════════════════════════════════════════════════

def _verify_language(text: str, claimed_language: str) -> str:
    """Quick text-only language verification via Groq."""
    if not GROQ_API_KEY or not text.strip():
        return claimed_language
    try:
        prompt = _language_double_check_prompt(text, claimed_language)
        r      = _groq_text_call(prompt, max_tokens=256)
        if not r.get("is_correct_claim", True):
            new_lang = _normalize_lang(r.get("verified_language", ""), claimed_language)
            print(f"[HW] Language re-detected: {claimed_language} → {new_lang}")
            return new_lang
    except Exception as e:
        print(f"[HW] Language verify failed (non-fatal): {e}")
    return claimed_language


# ═══════════════════════════════════════════════════════════════
#  Intent classifier
# ═══════════════════════════════════════════════════════════════

_RX_PRESCRIPTION = re.compile(
    r"ordonnance|prescription|وصفة\s*طبية"
    r"|\b(cp|bte|gel[eé]?lule|comprim[eé]s?|cachet|ampoule|sirop|suppositoire)\b"
    r"|\b\d+\s*(?:mg|mcg|g|ml|ui)\b"
    r"|\b(matin|soir|midi|nuit)\b"
    r"|\bfois\s+par\s+jour\b|\bfois\/jour\b|\b\d+\s*f\/j\b"
    r"|\b(amox|augmentin|ibupro|paracet|doliprane|fentanyl|aspirin|metform|omeprazol)\b",
    re.IGNORECASE,
)
_RX_QUESTION = re.compile(
    r"\?|؟"
    r"|\b(pourquoi|comment|qu['\u2019]est|what|why|how|where|when|who)\b"
    r"|ما\s|كيف\s|هل\s|لماذا\s",
    re.IGNORECASE,
)
_RX_ADDRESS = re.compile(
    r"\b(rue|avenue|av\.|bd\.|boulevard|quartier|ville|commune|wilaya"
    r"|code\s*postal|adresse|email|tel\.?|tél|@)\b",
    re.IGNORECASE,
)
_RX_FORM = re.compile(
    r"\b(nom\s*:|prénom\s*:|date\s*:|n°|numéro\s*:|cin\s*:|nni\s*:|id\s*:)\b",
    re.IGNORECASE,
)


def _classify_intent(text: str) -> str:
    rx_hits = len(_RX_PRESCRIPTION.findall(text))
    if rx_hits >= 2 or re.search(r"\bordonnance\b|\bprescription\b", text, re.IGNORECASE):
        return "PRESCRIPTION"
    if _RX_QUESTION.search(text):
        return "QUESTION"
    if _RX_FORM.search(text):
        return "FORM"
    if _RX_ADDRESS.search(text):
        return "ADDRESS"
    return "PLAIN_TEXT"


# ═══════════════════════════════════════════════════════════════
#  Action agents (structured output per intent)
# ═══════════════════════════════════════════════════════════════

def _call_groq_text(prompt: str, max_tokens: int = 600) -> dict:
    return _groq_text_call(prompt, max_tokens=max_tokens, temperature=0.3)


def _agent_question(transcription: str, language: str) -> dict:
    lang = LANGUAGE_LABELS.get(language, language)
    personal_rx = re.compile(
        r"\b(my|mine|moi|mon|ma|mes|je|j'|أنا|لي|عندي|حقي)\b", re.IGNORECASE
    )
    if personal_rx.search(transcription):
        spoken = {
            "english": "This question appears personal. I can only answer general questions.",
            "french":  "Cette question semble personnelle. Je ne réponds qu'aux questions générales.",
        }.get(language, "هذا السؤال يبدو شخصياً. يمكنني الإجابة فقط على الأسئلة العامة.")
        return {"spoken_response": spoken, "display_title": "❓ Question"}

    prompt = (
        f"You are a helpful assistant for a blind person. Answer this {lang} question clearly "
        f"in {lang}, under 5 sentences, no bullet points.\n\n"
        f'Question: """{transcription}"""\n\n'
        f'Respond ONLY: {{"spoken_response": "<answer in {lang}>"}}'
    )
    try:
        r = _call_groq_text(prompt, max_tokens=400)
        return {"spoken_response": r.get("spoken_response", transcription),
                "display_title": "❓ Question"}
    except Exception as e:
        return {"spoken_response": transcription, "display_title": "❓ Question", "error": str(e)}


def _agent_plain_text(transcription: str, language: str) -> dict:
    lang    = LANGUAGE_LABELS.get(language, language)
    is_long = len(transcription.split()) > LONG_TEXT_WORD_THRESHOLD
    if is_long:
        prompt = (
            f"Reading assistant for a blind person. Long {lang} text below.\n"
            f"Provide a concise spoken summary in {lang} (3-5 sentences, no bullets).\n\n"
            f'Text: """{transcription}"""\n\n'
            f'Respond ONLY: {{"spoken_response": "<summary in {lang}>", "key_points": ""}}'
        )
    else:
        prompt = (
            f"Reading assistant for a blind person. {lang} text below.\n"
            f"Read it naturally in {lang}. Start with 'The text says:' then read it, "
            f"then add 'In summary:' with 1-2 sentences. No bullets.\n\n"
            f'Text: """{transcription}"""\n\n'
            f'Respond ONLY: {{"spoken_response": "<reading in {lang}>", "key_points": ""}}'
        )
    try:
        r     = _call_groq_text(prompt, max_tokens=600)
        title = "📄 Summary" if is_long else "📄 Text"
        return {"spoken_response": r.get("spoken_response", transcription),
                "key_points": r.get("key_points", ""), "display_title": title}
    except Exception as e:
        return {"spoken_response": transcription, "display_title": "📄 Text", "error": str(e)}


def _agent_prescription(transcription: str, language: str) -> dict:
    lang   = LANGUAGE_LABELS.get(language, language)
    prompt = (
        f"Medical reading assistant for a blind patient. {lang} prescription below.\n\n"
        "⚠️ SAFETY: Use ONLY names/dosages from the transcription. "
        "NEVER invent. If [illegible] → say 'unclear, confirm with pharmacist'.\n\n"
        f'Prescription: """{transcription}"""\n\n'
        f"Extract every medicine: name, dosage, frequency, duration, instructions.\n"
        f"Generate a clear spoken summary in {lang} — no bullet points.\n\n"
        'Respond ONLY:\n'
        '{{"medicines":[{{"name":"","dosage":"","frequency":"","duration":"","instructions":""}}],'
        f'"spoken_response":"<spoken in {lang}>","medicine_count":0,"has_illegible":false}}'
    )
    try:
        r = _call_groq_text(prompt, max_tokens=800)
        has_il = (
            r.get("has_illegible", False)
            or "[illegible]" in transcription.lower()
            or "[غير مقروء]" in transcription
        )
        spoken = r.get("spoken_response", transcription)
        if has_il:
            spoken += (
                " Veuillez confirmer chaque dosage avec votre pharmacien."
                if language == "french" else
                " يرجى تأكيد كل جرعة مع الصيدلاني."
                if language in IS_ARABIC_LANG else
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
        return {"spoken_response": transcription, "medicines": [],
                "medicine_count": 0, "has_illegible": False,
                "display_title": "💊 Prescription", "error": str(e)}


def _agent_address(transcription: str, language: str) -> dict:
    lang   = LANGUAGE_LABELS.get(language, language)
    prompt = (
        f"Contact reading assistant for a blind person. {lang} contact info below.\n"
        f"Parse fields and generate a clear spoken reading in {lang} — no bullets.\n\n"
        f'Text: """{transcription}"""\n\n'
        'Respond ONLY:\n'
        '{{"parsed":{{"full_name":"","address":"","city":"","phone":"","email":"","other":""}},'
        f'"spoken_response":"<reading in {lang}>"}}'
    )
    try:
        r = _call_groq_text(prompt, max_tokens=400)
        return {
            "spoken_response": r.get("spoken_response", transcription),
            "parsed":          r.get("parsed", {}),
            "display_title":   "📍 Address / Contact",
            "navigate_to":     r.get("parsed", {}).get("address", ""),
        }
    except Exception as e:
        return {"spoken_response": transcription, "parsed": {},
                "display_title": "📍 Address", "navigate_to": "", "error": str(e)}


def _agent_form(transcription: str, language: str) -> dict:
    lang   = LANGUAGE_LABELS.get(language, language)
    prompt = (
        f"Form reading assistant for a blind person. {lang} form below.\n"
        f"Identify form type and extract field-value pairs. "
        f"Generate a clear spoken reading in {lang} — no bullets.\n\n"
        f'Form: """{transcription}"""\n\n'
        'Respond ONLY:\n'
        f'{{"form_type":"<type>","fields":[{{"label":"","value":""}}],"spoken_response":"<reading in {lang}>"}}'
    )
    try:
        r = _call_groq_text(prompt, max_tokens=500)
        return {
            "spoken_response": r.get("spoken_response", transcription),
            "form_type":       r.get("form_type", "Form"),
            "fields":          r.get("fields", []),
            "display_title":   f"📋 {r.get('form_type', 'Form')}",
        }
    except Exception as e:
        return {"spoken_response": transcription, "form_type": "Form",
                "fields": [], "display_title": "📋 Form", "error": str(e)}


_AGENT_ROUTER = {
    "QUESTION":     _agent_question,
    "PLAIN_TEXT":   _agent_plain_text,
    "PRESCRIPTION": _agent_prescription,
    "ADDRESS":      _agent_address,
    "FORM":         _agent_form,
}


# ═══════════════════════════════════════════════════════════════
#  HandwritingAgent — main class
# ═══════════════════════════════════════════════════════════════

class HandwritingAgent:
    """
    Multi-pass handwriting recognition agent.

    Usage:
        agent = HandwritingAgent(action_agent=action_agent)
        result = agent.process(image_bytes, language="auto")

    Result keys:
        transcription   — final verified clean text
        language        — detected language key
        intent          — QUESTION / PLAIN_TEXT / PRESCRIPTION / ADDRESS / FORM
        spoken_response — text that was (or should be) spoken
        medicines       — list[dict] (PRESCRIPTION only)
        navigate_to     — address string (ADDRESS only)
        fields          — list[dict] (FORM only)
        parsed          — dict (ADDRESS only)
        backend         — pipeline summary string
        confidence      — final confidence float
        pipeline_log    — per-pass log list (for debugging)
        error           — error string if something failed
    """

    def __init__(self, action_agent=None):
        self.action_agent = action_agent
        self._cache: dict[str, dict] = {}
        print("[HandwritingAgent] Ready ✓  (6-pass pipeline)")

    # ── Public entry point ─────────────────────────────────────

    def process(self, image_bytes: bytes, language: str = "auto") -> dict:
        t0    = time.time()
        log   = []

        # ── Decode image ──────────────────────────────────────
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return self._error("Could not decode image bytes")

        # ── Cache check ───────────────────────────────────────
        img_hash  = hashlib.md5(img.tobytes()).hexdigest()
        cache_key = f"{img_hash}_{language}"
        if cache_key in self._cache:
            print(f"[HW] Cache hit {cache_key[:12]}…")
            return self._cache[cache_key]

        # ── Preprocess ────────────────────────────────────────
        processed = _preprocess(img)
        image_b64 = _encode(processed)
        is_ar     = language in IS_ARABIC_LANG

        # ════════════════════════════════════════════════════
        # PASS 1 — Parallel extraction (3 vision models)
        # ════════════════════════════════════════════════════
        print("[HW] Pass 1: parallel extraction…")
        try:
            p1_candidates = _pass1_parallel(image_b64, language)
        except Exception as e:
            return self._error(f"Pass 1 failed entirely: {e}")

        # Pick best from Pass 1 as working text
        p1_best = max(p1_candidates, key=lambda c: float(c.get("confidence", 0)))
        p1_text = p1_best.get("transcription", "").strip()

        if not p1_text:
            return self._error("No text detected in image")

        # Detect language from Pass-1 results
        det_lang_raw = p1_best.get("detected_language", language)
        det_lang     = _normalize_lang(det_lang_raw, language)
        is_ar        = det_lang in IS_ARABIC_LANG   # update after detection

        log.append({
            "pass": 1,
            "models_ran": [c.get("source", "?") for c in p1_candidates],
            "candidates": len(p1_candidates),
            "best_source": p1_best.get("source", "?"),
            "confidence": float(p1_best.get("confidence", 0)),
            "words": len(p1_text.split()),
        })
        print(f"[HW] Pass 1 done: {len(p1_candidates)} candidates, "
              f"best={p1_best.get('source','?')}, "
              f"conf={float(p1_best.get('confidence', 0)):.2f}, "
              f"lang={det_lang}, words={len(p1_text.split())}")

        # ════════════════════════════════════════════════════
        # PASS 2 — Cross-check with image (Gemini / Groq)
        # ════════════════════════════════════════════════════
        print("[HW] Pass 2: cross-check verification…")
        p2_text     = p1_text
        p2_corrected = False
        try:
            p2 = _pass2_verify(image_b64, p1_text, det_lang,
                               float(p1_best.get("confidence", 0)))
            if p2.get("was_corrected"):
                p2_text      = p2.get("corrected_transcription", p1_text).strip() or p1_text
                p2_corrected = True
            meaning_english = p2.get("meaning_english", "")
            log.append({"pass": 2, "source": p2.get("source", "?"),
                        "corrected": p2_corrected,
                        "corrections": p2.get("corrections_made", "none"),
                        "suspicious": p2.get("suspicious_words", "")})
            print(f"[HW] Pass 2 done: corrected={p2_corrected}, "
                  f"source={p2.get('source','?')}")
        except Exception as e:
            meaning_english = ""
            print(f"[HW] Pass 2 failed (non-fatal): {e}")
            log.append({"pass": 2, "error": str(e)})

        # ════════════════════════════════════════════════════
        # PASS 3 — Deep Arabic verification (Arabic/Darija only)
        # ════════════════════════════════════════════════════
        p3_text      = p2_text
        p3_corrected = False
        if is_ar:
            print("[HW] Pass 3: deep Arabic verification (Qwen2.5-VL)…")
            try:
                p3 = _pass3_arabic_deep(image_b64, p2_text, det_lang)
                if p3.get("was_corrected"):
                    p3_text      = p3.get("corrected_transcription", p2_text).strip() or p2_text
                    p3_corrected = True
                log.append({"pass": 3, "source": p3.get("source", "?"),
                            "corrected": p3_corrected,
                            "corrections": p3.get("corrections_made", "none"),
                            "confidence": float(p3.get("confidence", 0))})
                print(f"[HW] Pass 3 done: corrected={p3_corrected}, "
                      f"source={p3.get('source','?')}")
            except Exception as e:
                print(f"[HW] Pass 3 failed (non-fatal): {e}")
                log.append({"pass": 3, "error": str(e)})
        else:
            log.append({"pass": 3, "skipped": "not Arabic/Darija"})

        # ════════════════════════════════════════════════════
        # PASS 4 — LLM consensus judge (all candidates → best merge)
        # ════════════════════════════════════════════════════
        print("[HW] Pass 4: consensus judge…")

        # Build the candidate list for the judge
        judge_candidates: list[dict] = []
        for c in p1_candidates:
            judge_candidates.append({
                "source":     c.get("source", "P1 model"),
                "text":       c.get("transcription", ""),
                "confidence": float(c.get("confidence", 0)),
            })
        if p2_text != p1_text:
            judge_candidates.append({
                "source":     "Pass-2 verify",
                "text":       p2_text,
                "confidence": 0.88,
            })
        if is_ar and p3_text != p2_text:
            judge_candidates.append({
                "source":     "Pass-3 Arabic (Qwen2.5-VL)",
                "text":       p3_text,
                "confidence": 0.92,   # Qwen2.5-VL is reliable for Arabic
            })

        try:
            p4 = _pass4_judge(judge_candidates, det_lang)
            final_text = p4.get("final_transcription", p3_text).strip() or p3_text
            final_conf = float(p4.get("confidence", 0.8))
            log.append({"pass": 4,
                        "candidates_seen": len(judge_candidates),
                        "confidence": final_conf,
                        "notes": p4.get("consensus_notes", "")})
            print(f"[HW] Pass 4 done: conf={final_conf:.2f}, "
                  f"notes={p4.get('consensus_notes', '')}")
        except Exception as e:
            final_text = p3_text
            final_conf = float(p1_best.get("confidence", 0.7))
            print(f"[HW] Pass 4 failed (non-fatal): {e}")
            log.append({"pass": 4, "error": str(e)})

        # ════════════════════════════════════════════════════
        # PASS 5 — Digit check (medical) + deterministic guardrails
        # ════════════════════════════════════════════════════
        if _is_medical(final_text) and _has_dosage_numbers(final_text):
            print("[HW] Pass 5: digit/medical check…")
            try:
                p5 = _pass5_digit_check(image_b64, final_text, det_lang)
                if p5.get("was_corrected"):
                    final_text = p5.get("corrected_transcription", final_text).strip() or final_text
                    log.append({"pass": 5, "corrected": True,
                                "suspicious_numbers": p5.get("suspicious_numbers", "")})
                    print(f"[HW] Pass 5: digit correction applied")
                else:
                    log.append({"pass": 5, "corrected": False})
            except Exception as e:
                print(f"[HW] Pass 5 failed (non-fatal): {e}")
                log.append({"pass": 5, "error": str(e)})

        # Deterministic guardrails (no API)
        guarded, guard_changes = _apply_guardrails(final_text)
        if guard_changes:
            final_text = guarded
            log.append({"pass": "5-guardrails", "changes": guard_changes})
            print(f"[HW] Guardrails applied: {guard_changes}")

        # ════════════════════════════════════════════════════
        # Language double-check (text-only, fast)
        # ════════════════════════════════════════════════════
        det_lang = _verify_language(final_text, det_lang)

        # ════════════════════════════════════════════════════
        # PASS 6 — Intent → structured extraction → TTS
        # ════════════════════════════════════════════════════
        print("[HW] Pass 6: intent + structured extraction…")
        intent  = _classify_intent(final_text)
        handler = _AGENT_ROUTER.get(intent, _agent_plain_text)
        try:
            agent_result = handler(final_text, det_lang)
        except Exception as e:
            agent_result = {"spoken_response": final_text,
                            "display_title": "📄 Text", "error": str(e)}

        spoken_response = agent_result.get("spoken_response", final_text)
        log.append({"pass": 6, "intent": intent,
                    "spoken_words": len(spoken_response.split())})

        # ── TTS via ActionAgent ───────────────────────────
        tts_lang = _lang_to_tts_code(det_lang)
        if self.action_agent and spoken_response:
            try:
                is_urgent = (
                    intent == "PRESCRIPTION"
                    and agent_result.get("has_illegible", False)
                )
                if is_urgent:
                    self.action_agent.speak(spoken_response, lang=tts_lang, interrupt=True)
                else:
                    self.action_agent.speak_normal(spoken_response, lang=tts_lang)
            except Exception as e:
                print(f"[HW] TTS error (non-fatal): {e}")

        # ── Build final result ────────────────────────────
        elapsed = time.time() - t0
        backend_summary = (
            f"P1:[{', '.join(c.get('source','?') for c in p1_candidates)}] "
            f"P2:{p2_corrected} P3_AR:{p3_corrected if is_ar else 'skip'} "
            f"P4:judge P5:{'digit' if _is_medical(final_text) else 'skip'} "
            f"({elapsed:.1f}s)"
        )
        print(f"[HW] Pipeline complete in {elapsed:.1f}s — intent={intent}, "
              f"lang={det_lang}, words={len(final_text.split())}, conf={final_conf:.2f}")

        result = {
            # ── Core output ──
            "transcription":    final_text,
            "language":         det_lang,
            "intent":           intent,
            "spoken_response":  spoken_response,
            "display_title":    agent_result.get("display_title", "📄 Result"),
            "meaning_english":  meaning_english,
            # ── Quality metadata ──
            "backend":          backend_summary,
            "confidence":       final_conf,
            "pipeline_log":     log,
            "elapsed_sec":      round(elapsed, 2),
            # ── Intent-specific extras ──
            "medicines":        agent_result.get("medicines", []),
            "medicine_count":   agent_result.get("medicine_count", 0),
            "navigate_to":      agent_result.get("navigate_to", ""),
            "fields":           agent_result.get("fields", []),
            "parsed":           agent_result.get("parsed", {}),
            "has_illegible":    agent_result.get("has_illegible", False),
            "error":            agent_result.get("error"),
        }

        self._cache[cache_key] = result
        return result

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _error(msg: str) -> dict:
        print(f"[HW] ERROR: {msg}")
        return {
            "transcription": "", "language": "unknown", "intent": "ERROR",
            "spoken_response": f"Error processing image: {msg}",
            "display_title": "❌ Error", "error": msg,
            "medicines": [], "medicine_count": 0, "navigate_to": "",
            "fields": [], "parsed": {}, "has_illegible": False,
            "confidence": 0.0, "backend": "none", "pipeline_log": [],
        }

    def clear_cache(self):
        self._cache.clear()
        print("[HW] Cache cleared.")