"""
DialogueAgent.py — v6 (LLM-first, no rules)
─────────────────────────────────────────────────────────────────
Replaces the v5 rule-based intent extractor (~950 lines of Arabic
roots, French stopwords, regex rules, canonical-noun maps) with a
single LLM call using a tightly-scoped prompt and strict JSON
output schema.

Why this works:

  • Modern LLMs natively understand all 4 of our languages
    (English, French, Arabic, Tunisian Derja) — no separate
    keyword lists, no per-language code paths.

  • Whisper STT mishearings (نلقى → نرقى, تلفون → طالفون,
    téléphone → "telephone non") are trivially correctable
    by the LLM in context: "the user said 'I want to find my
    طالفوني' — that's clearly تلفون with a Whisper artifact,
    return target='تلفون'."

  • Entity extraction, language detection, intent classification,
    and noun canonicalization happen in ONE round-trip.

  • The pricing case for the old rule layer was "save LLM calls."
    But the bottleneck in this app is the VLM (vision), not the
    LLM (intent).  Intent runs ~once per voice command — at most
    ~50 calls/day on a free account.  Easily fits within free-tier
    rate limits across multiple providers.

What's left as plain code (NOT model-driven):

  1. YES/NO detection during confirmation flow — kept as a small
     lookup because we explicitly ask the user to say "YES" or
     "NO" in English to avoid Whisper mistranscribing Arabic
     إيه as "Hey".

  2. Confirmation state machine (pending action, language
     preservation across the yes/no turn).  This is application
     logic, not language understanding.

  3. Localized confirmation question templates (one short string
     per language per intent).  We don't need a model to translate
     "Did you ask me to find your phone?" — it's the same sentence
     every time.

The end-to-end flow:

  Whisper STT → text + lang_hint (from STT)
       │
       ▼
  process_text(text):
       │
       ├─ pending_action exists?
       │     └─ Yes → check yes/no, route accordingly
       │
       ├─ Single LLM call:
       │     "Extract intent (navigate|find|describe|other),
       │      language (en|fr|ar|tn), target_noun (clean form),
       │      and is_specific_destination (bool, navigate only).
       │      Correct any STT mishearings."
       │
       └─ Build localized confirmation question, store pending,
          return.
"""

from __future__ import annotations

import os
import json
import time
import re
import requests
from collections import deque
from typing import Optional

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("[DialogueAgent] faster-whisper not installed.")
    WHISPER_AVAILABLE = False


# ─── YES / NO matching ─────────────────────────────────────────
# Kept as code (not model) because we explicitly prompt the user
# to say YES or NO in English.  The model would just round-trip
# the same answer slower.
_YES_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure",
    "go", "do it", "confirm", "correct", "right", "exactly",
    "oui", "ouais", "d'accord", "vas-y",
})
_NO_WORDS = frozenset({
    "no", "nope", "nah", "cancel", "stop", "wrong", "abort", "not",
    "non", "annule",
})


def _is_confirmation(text: str) -> Optional[bool]:
    """Returns True for yes-like answers, False for no-like, None otherwise."""
    norm = text.strip().lower().rstrip(".!?,;:")
    norm = re.sub(r"[^a-zà-ÿ\s']", "", norm).strip()
    if not norm:
        return None
    tokens = norm.split()
    if norm in _YES_WORDS or any(t in _YES_WORDS for t in tokens):
        if any(t in _NO_WORDS for t in tokens):
            return None  # mixed signal → re-prompt
        return True
    if norm in _NO_WORDS or any(t in _NO_WORDS for t in tokens):
        return False
    return None


# ─── Localized confirmation templates ──────────────────────────
# These are the only multilingual strings.  Everything else (intent
# detection, entity extraction, language detection, mishearing
# correction) is delegated to the LLM.

_CONFIRM_FIND = {
    "en": 'Did you ask me to find your {entity}? Say YES to start, or NO to cancel.',
    "fr": 'Vous voulez que je trouve votre {entity} ? Dites YES pour commencer, ou NO pour annuler.',
    "ar": 'هل تريدني أن أبحث عن {entity}؟ قل YES للبدء، أو NO للإلغاء.',
    "tn": 'تحب نلقالك {entity}؟ قول YES باش نبدا، ولا NO باش نوقف.',
}

_CONFIRM_NAV_GENERIC = {
    "en": 'You said {entity}. Do you want the closest one, or a specific place? Say a name, or say closest.',
    "fr": 'Vous avez dit {entity}. Voulez-vous le plus proche, ou un endroit précis ? Dites un nom ou dites closest.',
    "ar": 'قلت {entity}. تريد الأقرب أم مكاناً محدداً؟ قل اسماً أو قل closest.',
    "tn": 'قلت {entity}. تحب القريب ولا مكان معين؟ قل الاسم ولا قل closest.',
}

_CONFIRM_SAVE_OBJ = {
    "en": 'You want me to save this as {entity}? Say YES to open the camera, or NO to cancel.',
    "fr": 'Vous voulez sauvegarder ceci comme {entity} ? Dites YES pour ouvrir la caméra, ou NO pour annuler.',
    "ar": 'تريد حفظ هذا كـ {entity}؟ قل YES لفتح الكاميرا، أو NO للإلغاء.',
    "tn": 'تحب نحفظ هذا كـ {entity}؟ قول YES باش نفتح الكاميرا، ولا NO باش نوقف.',
}

_CONFIRM_NAV_SPECIFIC = {
    "en": 'You want me to navigate to {entity}? Say YES to confirm, or NO to cancel.',
    "fr": 'Vous voulez aller à {entity} ? Dites YES pour confirmer, ou NO pour annuler.',
    "ar": 'تريد الذهاب إلى {entity}؟ قل YES للتأكيد، أو NO للإلغاء.',
    "tn": 'تحب نوصلك لـ {entity}؟ قول YES باش نتأكد، ولا NO باش نوقف.',
}

_CONFIRM_DESCRIBE = {
    "en": "Did you ask me to describe what's around you? Say YES to start, or NO to cancel.",
    "fr": "Vous voulez que je décrive ce qui vous entoure ? Dites YES pour commencer, ou NO pour annuler.",
    "ar": "هل تريدني أن أصف ما حولك؟ قل YES للبدء، أو NO للإلغاء.",
    "tn": "تحب نوصفلك اللي قدامك؟ قول YES باش نبدا، ولا NO باش نوقف.",
}

_CANCELLED = {
    "en": "Okay, cancelled. What would you like to do?",
    "fr": "D'accord, annulé. Que voulez-vous faire ?",
    "ar": "حسناً، تم الإلغاء. ماذا تريد أن تفعل؟",
    "tn": "بيهي، توقفنا. شنو تحب نعمل؟",
}

_REPROMPT_YES_NO = {
    "en": "I didn't catch that. Please say YES or NO.",
    "fr": "Je n'ai pas compris. Dites YES ou NO.",
    "ar": "لم أفهم. من فضلك قل YES أو NO.",
    "tn": "ما فهمتش. عاود قول YES ولا NO.",
}

_DID_NOT_UNDERSTAND = {
    "en": "I didn't catch that clearly. You can say: take me to a place, find an object, or describe what's around me.",
    "fr": "Je n'ai pas bien compris. Dites par exemple : amène-moi quelque part, trouve un objet, ou décris ce qui m'entoure.",
    "ar": "لم أفهم جيداً. يمكنك أن تقول مثلاً: خذني إلى مكان، ابحث عن شيء، أو صف ما حولي.",
    "tn": "ما فهمتش مليح. تنجم تقول مثلاً: روحني لمكان، لقّيلي حاجة، ولا وصفلي اللي قدامي.",
}


# ─── The single LLM prompt ─────────────────────────────────────
# This is the heart of the agent.  Tightly-scoped, schema-strict,
# multilingual-aware.  Tested against all the Whisper hallucination
# cases we used to handle with regex.

_INTENT_SYSTEM_PROMPT = """\
You are the intent extractor for a voice assistant for blind users.

The user speaks one of: English (en), French (fr), Modern Standard
Arabic (ar), or Tunisian Derja (tn).  Their voice is transcribed by
Whisper, which can produce SMALL ERRORS — letter swaps, dropped
phonemes, or words that don't quite exist.  When you see something
that LOOKS LIKE a real word with one or two off characters, treat it
as that real word.  Examples of typical Whisper artifacts:

  - "I want to find my telephone" → target="phone"
  - "أريد أن أجدنا نظراتي"        → target="نظارات" (glasses)
  - "نحب نلقى تالفوني"            → target="تلفون"  (phone, Tunisian)
  - "نحب نرقى طالفوني"            → target="تلفون"  (heavy STT noise)
  - "trouve mon téléfone"         → target="téléphone"
  - "أين هاتفني"                  → target="هاتف"   (my phone)

Return ONE JSON object with this EXACT schema (no extra keys, no
prose, no markdown fences):

{
  "intent": "navigate" | "find" | "describe" | "other",
  "language": "en" | "fr" | "ar" | "tn",
  "target": "<single canonical noun, in the user's language>",
  "is_specific_destination": true | false,
  "confidence": 0.0-1.0
}

Rules:

- intent="find"     when the user wants to LOCATE an object they own
                    or that's nearby (phone, keys, glasses, wallet,
                    remote, mug, bag, …).  target is the object.

- intent="navigate" when the user wants to GO somewhere or wants
                    DIRECTIONS (pharmacy, hospital, café, an address,
                    "take me to X", "where is X", "I want to go to X").
                    target is the destination.

- intent="describe" when the user asks WHAT IS AROUND them, what's
                    in front, what's in the room, or wants a scene
                    description.  target is "" (empty string).

- intent="other"    when none of the above clearly applies, or the
                    utterance is too short / unintelligible.

- language: detect from the utterance.  Tunisian Derja is Arabic
            script with North-African dialect words like "نحب", "هك",
            "باش", "شنو", "كيفاش", "برشا".  Modern Standard Arabic
            uses "أريد", "كيف", "أين", "كثير".  When in doubt between
            ar and tn for an Arabic utterance, prefer "tn" if any
            Derja markers are present, else "ar".

- target: clean, canonical, in the USER'S language.  Strip pronouns
          ("my phone" → "phone", "هاتفي" → "هاتف", "تلفوني" → "تلفون").
          Correct Whisper mishearings.  For "describe" use "".

- is_specific_destination: navigate-intent only.  TRUE when the user
          named a specific place ("Carrefour Lac 2", "Rue de Marseille",
          "the Habib Bourguiba airport").  FALSE for generic POIs
          ("a pharmacy", "any café", "the closest hospital").
          Always FALSE for non-navigate intents.

- confidence: how sure you are.  0.9+ for clear utterances, 0.6-0.8
          for noisy or ambiguous, <0.5 for "I'm guessing".

Output ONLY the JSON object.  Nothing else."""


_FEW_SHOT_EXAMPLES = [
    # (user_utterance, expected_json)
    ('I want to find my phone',
     '{"intent":"find","language":"en","target":"phone",'
     '"is_specific_destination":false,"confidence":0.97}'),

    ('je veux trouver mon téléphone',
     '{"intent":"find","language":"fr","target":"téléphone",'
     '"is_specific_destination":false,"confidence":0.97}'),

    ('نحب نلقى تلفوني',
     '{"intent":"find","language":"tn","target":"تلفون",'
     '"is_specific_destination":false,"confidence":0.95}'),

    ('نحب نرقى طالفوني',  # heavy Whisper noise
     '{"intent":"find","language":"tn","target":"تلفون",'
     '"is_specific_destination":false,"confidence":0.7}'),

    ('أريد أن أجد هاتفي',
     '{"intent":"find","language":"ar","target":"هاتف",'
     '"is_specific_destination":false,"confidence":0.95}'),

    ('take me to a pharmacy',
     '{"intent":"navigate","language":"en","target":"pharmacy",'
     '"is_specific_destination":false,"confidence":0.95}'),

    ('roohni l Carrefour Lac 2',
     '{"intent":"navigate","language":"tn","target":"Carrefour Lac 2",'
     '"is_specific_destination":true,"confidence":0.9}'),

    ('describe what is around me',
     '{"intent":"describe","language":"en","target":"",'
     '"is_specific_destination":false,"confidence":0.95}'),

    ('وصفلي اللي قدامي',
     '{"intent":"describe","language":"tn","target":"",'
     '"is_specific_destination":false,"confidence":0.95}'),

    ('hey',
     '{"intent":"other","language":"en","target":"",'
     '"is_specific_destination":false,"confidence":0.2}'),
]


# ─── Generic POIs (used to decide which confirmation question to ask) ─
# Tiny list — the LLM tells us is_specific_destination, but we still
# use this as a backup hint for which confirmation template to pick.
_GENERIC_POI_HINTS = frozenset({
    "pharmacy", "pharmacie", "صيدلية",
    "hospital", "hôpital", "مستشفى",
    "café", "cafe", "coffee", "مقهى",
    "restaurant", "مطعم",
    "supermarket", "supermarché", "سوبر ماركت",
    "bank", "banque", "بنك",
    "atm", "distributeur",
    "bakery", "boulangerie", "مخبزة",
    "school", "école", "مدرسة",
    "mosque", "mosquée", "مسجد", "جامع",
    "church", "église", "كنيسة",
    "gas station", "station service",
    "police", "post office", "poste",
    "park", "parc", "حديقة",
})


# ─── Agent ─────────────────────────────────────────────────────

class DialogueAgent:

    _MAX_CONTEXT = 4

    def __init__(
        self,
        api_key: str,
        whisper_model_size:   str = "small",
        whisper_device:       str = "cpu",
        whisper_compute_type: str = "int8",
        llm_model:            str = "meta-llama/llama-3.3-70b-instruct:free",
        llm_fallbacks:        Optional[list] = None,
        google_api_key:       str = "",
    ):
        self.api_key        = api_key
        self.google_api_key = (google_api_key or "").strip()
        self.llm_model      = llm_model
        self.llm_fallbacks  = llm_fallbacks or [
            # Working free models as of 2026 — ordered by reliability
            "google/gemma-3-27b-it:free",           # solid, rarely 429s
            "mistralai/mistral-7b-instruct:free",   # tiny, fast, good JSON
            "meta-llama/llama-3.1-8b-instruct:free",# sometimes 404s — last OR attempt
        ]
        self.llm_url    = "https://openrouter.ai/api/v1/chat/completions"
        self.gemini_url = ("https://generativelanguage.googleapis.com/v1beta/models")

        # Models confirmed dead/broken — never retry them
        self._dead_models: set  = set()
        self._always_skip: set  = {
            "openrouter/free",                   # always returns empty content
            "nvidia/nemotron-nano-9b-v2:free",   # always returns empty content
            "google/gemma-3-12b-it:free",        # consistent 400 errors
        }

        self._context: deque    = deque(maxlen=self._MAX_CONTEXT)
        self.pending_action: Optional[dict] = None
        self._last_stt_lang: str = "en"

        # Groq — fast text-only fallback (key shared with HandwritingAgent)
        try:
            from config import GROQ_API_KEY as _gk
            self._groq_key = (_gk or "").strip()
        except Exception:
            self._groq_key = ""

        self.whisper = None
        if WHISPER_AVAILABLE:
            print(f"[DialogueAgent] Loading Whisper '{whisper_model_size}'…")
            self.whisper = WhisperModel(
                whisper_model_size, device=whisper_device,
                compute_type=whisper_compute_type)
            print("[DialogueAgent] Whisper ready.")

    # ── Public API ────────────────────────────────────────────

    def process_audio(self, wav_path: str) -> dict:
        text = self._transcribe(wav_path)
        if not text:
            return self._fallback("", reason="empty STT")
        return self.process_text(text)

    def process_text(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return self._fallback(text, reason="empty input")

        # ── 1. Confirmation reply? Handle yes/no first ──
        if self.pending_action is not None:
            return self._handle_confirmation_reply(text)

        # ── 1b. Local intent detection — NO LLM needed ────────────────
        # Covers the most common voice commands so the app works even
        # when every external API is down or rate-limited.
        import unicodedata as _ud
        _t_lower = text.lower().strip()

        # Normalise Arabic: unify alef variants + strip harakat diacritics
        _t_ar = _ud.normalize("NFC", text)
        _t_ar = re.sub(r"[\u064B-\u065F\u0670]", "", _t_ar)  # strip harakat
        _t_ar = re.sub(r"[إأآا]", "ا", _t_ar)                # unify alef

        # Detect language + Darija vs MSA
        _ar_chars   = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        _is_arabic  = _ar_chars > len(text) * 0.3
        _darija_markers = ["نحب", "باش", "هك", "كيفاش", "شنو", "برشا",
                           "ولا", "قرالي", "روحني", "وصفلي", "لقيلي", "لقّيلي"]
        if _is_arabic:
            _lang_guess = "tn" if any(m in text for m in _darija_markers) else "ar"
        elif any(w in _t_lower for w in ["le ", "la ", "les ", "lis ", "lire",
                                          "amène", "emmène", "trouve", "décris"]):
            _lang_guess = "fr"
        else:
            _lang_guess = "en"

        # ── A. READ / HANDWRITING ──────────────────────────────────────
        _READ_LATIN = [
            "read the text", "read text", "read this", "read that",
            "scan the text", "scan text", "what does it say",
            "what does this say", "what does that say",
            "read handwriting", "read the handwriting",
            "read the note", "read the letter", "read the sign",
            "read the prescription", "read prescription",
            "read this text", "read that text",
            "lis le texte", "lire le texte", "lis ça", "lire ça",
            "lis ce texte", "lire ce texte", "qu'est-ce que ça dit",
            "qu'est-ce qu'il y a écrit", "lire l'ordonnance",
            "lis l'ordonnance", "lis la note", "lire la note",
        ]
        # Arabic: match on normalised string (_t_ar) — harakat & alef insensitive
        _READ_ARABIC = [
            "اقرا النص", "اقرا لي", "اقرالي",       # covers "اقرأ لي نص"
            "اقرا هذا",  "اقرا الوصفة",
            "شنو مكتوب", "شنوة مكتوب",
            "ايش مكتوب",
            "اقرا",   # broad: any "read" command in Arabic
            "قرالي",  # Darija: "read it to me"
            "قرا لي",
        ]
        _is_read = (
            any(kw in _t_lower for kw in _READ_LATIN)
            or any(kw in _t_ar   for kw in _READ_ARABIC)
        )
        if _is_read:
            return {
                "intent": "read_handwriting", "entities": [],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── B. DESCRIBE (scene / environment) ─────────────────────────
        _DESC_LATIN = [
            "describe", "what's around", "what is around",
            "what do you see", "what can you see", "look around",
            "what's in front", "what is in front", "what's here",
            "décris", "qu'est-ce qu'il y a", "qu'est-ce que tu vois",
            "regarde autour", "dis-moi ce que tu vois",
        ]
        _DESC_ARABIC = [
            "وصفلي", "وصف لي", "صفلي",
            "شنو قدامي", "شنو حواليا", "شنو في قدامي",
            "ايش امامي", "ما الذي امامي", "ما الذي حولي",
            "صف ما حولي", "صف ما امامي",
        ]
        _is_describe = (
            any(kw in _t_lower for kw in _DESC_LATIN)
            or any(kw in _t_ar   for kw in _DESC_ARABIC)
        )
        if _is_describe:
            _clar_q = _CONFIRM_DESCRIBE.get(_lang_guess, _CONFIRM_DESCRIBE["en"])
            self.pending_action = {
                "intent": "env_action", "entities": [],
                "mode": "DESCRIBE", "language": _lang_guess,
            }
            return {
                "intent": "env_action", "entities": [],
                "language": _lang_guess, "confidence": 0.95,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": _clar_q,
                "awaiting_confirmation": True, "confirmed": False,
                "parse_error": False, "sub_intent": "describe",
            }

        # ── B1. DELETE LOCATION — "delete this location" / "forget home" ──
        _DEL_LOC_LATIN = [
            "delete this location", "delete location", "remove this location",
            "forget this location", "forget location", "forget home",
            "forget work", "delete home", "delete work",
            "supprime cet endroit", "oublie cet endroit",
        ]
        _DEL_LOC_ARABIC = [
            "احذف هذا المكان", "امسح هذا المكان", "انسى هذا المكان",
        ]
        _is_del_loc = (
            any(kw in _t_lower for kw in _DEL_LOC_LATIN)
            or any(kw in _t_ar   for kw in _DEL_LOC_ARABIC)
        )
        if _is_del_loc:
            # Extract label if specified (e.g. "forget home" → "home")
            _del_label = ""
            _del_triggers = ["forget ", "delete ", "remove ", "supprime ", "oublie "]
            for tr in _del_triggers:
                if tr in _t_lower:
                    _after = _t_lower.split(tr, 1)[-1].strip(" ?.,!")
                    if _after not in ("this location", "location", "cet endroit", "endroit"):
                        _del_label = _after
                    break
            return {
                "intent": "delete_location", "entities": [_del_label] if _del_label else [],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── C0. SAVE LOCATION — "save this location as home" ─────────
        # Must come BEFORE location query so "save my location as home"
        # doesn't get caught by "my location" in the query list.
        _SAVE_LOC_LATIN = [
            "save this location as", "save my location as",
            "remember this location as", "remember this place as",
            "set this as my", "mark this as",
            "enregistre cet endroit comme", "sauvegarde ma position comme",
        ]
        _SAVE_LOC_ARABIC = [
            "احفظ هذا المكان كـ", "احفظ موقعي كـ",
            "سجل هذا المكان كـ", "خلي هذا المكان",
        ]
        _is_save_loc = (
            any(kw in _t_lower for kw in _SAVE_LOC_LATIN)
            or any(kw in _t_ar   for kw in _SAVE_LOC_ARABIC)
        )
        if _is_save_loc:
            _loc_label = ""
            for kw in sorted(_SAVE_LOC_LATIN, key=len, reverse=True):
                if kw in _t_lower:
                    _loc_label = _t_lower.split(kw, 1)[-1].strip(" ?.,!")
                    break
            if not _loc_label:
                for kw in sorted(_SAVE_LOC_ARABIC, key=len, reverse=True):
                    if kw in _t_ar:
                        _loc_label = _t_ar.split(kw, 1)[-1].strip(" ?.,!")
                        break
            if not _loc_label:
                _loc_label = "home"
            # ── Whisper correction for common location names ──
            _LOC_CORRECTIONS = {
                "hell": "home", "hom": "home", "hone": "home",
                "work": "work", "wore": "work", "wok": "work",
                "school": "school", "skool": "school",
            }
            _loc_label = _LOC_CORRECTIONS.get(_loc_label.lower(), _loc_label)
            return {
                "intent": "save_location", "entities": [_loc_label],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── C0b. SAVE OBJECT — "save this as my keys" / "remember this as John" ──
        _SAVE_OBJ_LATIN = [
            "save this as", "remember this as", "memorize this as",
            "save this object as", "remember this object as",
            "save this person as", "remember this person as",
            "enregistre ça comme", "mémorise ça comme",
            "enregistre ceci comme", "souviens-toi de ça comme",
        ]
        _SAVE_OBJ_ARABIC = [
            "احفظ هذا كـ", "تذكر هذا كـ", "سجل هذا كـ",
            "احفظ هذا ك", "خلي هذا كـ",
        ]
        _is_save_obj = (
            any(kw in _t_lower for kw in _SAVE_OBJ_LATIN)
            or any(kw in _t_ar   for kw in _SAVE_OBJ_ARABIC)
        )
        if _is_save_obj:
            _obj_name = ""
            for kw in sorted(_SAVE_OBJ_LATIN, key=len, reverse=True):
                if kw in _t_lower:
                    _obj_name = _t_lower.split(kw, 1)[-1].strip(" ?.,!")
                    break
            if not _obj_name:
                for kw in sorted(_SAVE_OBJ_ARABIC, key=len, reverse=True):
                    if kw in _t_ar:
                        _obj_name = _t_ar.split(kw, 1)[-1].strip(" ?.,!")
                        break
            if not _obj_name:
                _obj_name = "object"
            # Strip possessive: "my keys" → "keys"
            _obj_name = re.sub(r"^(my|your|the|mon|ma|mes|le|la|les)\s+",
                               "", _obj_name).strip()
            _clar_q = _CONFIRM_SAVE_OBJ.get(
                _lang_guess, _CONFIRM_SAVE_OBJ["en"]).format(entity=_obj_name)
            self.pending_action = {
                "intent": "save_object", "entities": [_obj_name],
                "mode": "SAVE_OBJECT", "language": _lang_guess,
            }
            return {
                "intent": "save_object", "entities": [_obj_name],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": _clar_q,
                "awaiting_confirmation": True, "confirmed": False,
                "parse_error": False,
            }

        # ── C0c. SAVE PERSON — "this is my friend Ahmed" ──────────────────
        _REL_MAP = {
            "friend": "friend", "ami": "friend", "amie": "friend",
            "brother": "brother", "frère": "brother",
            "sister": "sister", "sœur": "sister",
            "mother": "mother", "mère": "mother", "mom": "mother",
            "father": "father", "père": "father", "dad": "father",
            "family": "family", "famille": "family",
            "colleague": "colleague", "collègue": "colleague",
            "husband": "husband", "mari": "husband",
            "wife": "wife", "femme": "wife",
            "son": "son", "fils": "son",
            "daughter": "daughter", "fille": "daughter",
            "uncle": "uncle", "aunt": "aunt",
            "cousin": "cousin",
        }
        # ── C0c-i. SAVE MY FACE — "save my face" / "this is me" ──────────
        _SAVE_SELF_PHRASES = [
            "save my face", "remember my face", "remember me",
            "this is me", "that's me", "it's me",
            "save me", "add my face",
            "enregistre mon visage", "c'est moi", "souviens-toi de moi",
            "احفظ وجهي", "هذا أنا", "تذكرني",
        ]
        _is_save_self = any(ph in _t_lower or ph in _t_ar
                            for ph in _SAVE_SELF_PHRASES)
        if _is_save_self:
            self.pending_action = {
                "intent": "save_person", "entities": ["__SELF__", "self"],
                "mode": "SAVE_PERSON", "language": _lang_guess,
            }
            return {
                "intent": "save_person", "entities": ["__SELF__", "self"],
                "language": _lang_guess, "confidence": 0.98,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        _SAVE_PERSON_PHRASES = [
            "this is my friend", "this is my brother", "this is my sister",
            "this is my mother", "this is my father", "this is my family",
            "this is my colleague", "this is my husband", "this is my wife",
            "this is my son", "this is my daughter", "this is my uncle",
            "this is my aunt", "this is my cousin", "this is my mom",
            "this is my dad",
            "c'est mon ami", "c'est mon amie", "c'est ma sœur", "c'est mon frère",
            "c'est mon père", "c'est ma mère", "c'est ma famille",
            "c'est mon collègue", "c'est mon mari", "c'est ma femme",
        ]
        _SAVE_PERSON_ARABIC = [
            "هذا صديقي", "هذه صديقتي", "هذا أخي", "هذه أختي",
            "هذا أبي", "هذه أمي", "هذا ابني", "هذه ابنتي",
            "هذا عمي", "هذه عمتي", "هذا زوجي", "هذه زوجتي",
            "هذا زميلي", "تعرف على", "احفظ وجه",
        ]
        _is_save_person = (
            any(ph in _t_lower for ph in _SAVE_PERSON_PHRASES)
            or any(kw in _t_ar   for kw in _SAVE_PERSON_ARABIC)
        )
        if _is_save_person:
            # Extract relationship
            _rel = "person"
            for word, mapped in _REL_MAP.items():
                if word in _t_lower:
                    _rel = mapped
                    break
            # Extract name: text after the matched phrase
            _pname = ""
            for ph in sorted(_SAVE_PERSON_PHRASES, key=len, reverse=True):
                if ph in _t_lower:
                    _after = _t_lower.split(ph, 1)[-1].strip(" ,.:?!")
                    # Remove "named", "called", "his name is", "her name is"
                    _after = re.sub(
                        r"^(named?|called|his name is|her name is|,)\s*", "", _after).strip()
                    if _after:
                        _pname = _after
                    break
            if not _pname:
                for kw in sorted(_SAVE_PERSON_ARABIC, key=len, reverse=True):
                    if kw in _t_ar:
                        _pname = _t_ar.split(kw, 1)[-1].strip(" ,.:?!")
                        break
            if not _pname:
                _pname = "person"
            _pname = re.sub(r"^(my|the|a|mon|ma|mes|le|la|les)\s+", "", _pname).strip()
            # Take first word, strip any trailing punctuation (e.g. "Omar,")
            _pname = _pname.split()[0].strip(".,!?\"';:") if _pname else "person"

            self.pending_action = {
                "intent": "save_person", "entities": [_pname, _rel],
                "mode": "SAVE_PERSON", "language": _lang_guess,
            }
            # confirmed=True / awaiting=False → camera opens immediately,
            # no second "say YES" tap required (intent is unambiguous).
            return {
                "intent": "save_person", "entities": [_pname, _rel],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── C1. LOCATION QUERY — "where am I?" / "where is my location?" ──
        # Route directly to navigate (no camera, uses GPS + reverse geocode).
        _LOCATION_LATIN = [
            "where am i", "where are we", "where is my location",
            "my location", "current location", "what is my location",
            "what's my location", "where am i right now",
            "what street am i on", "which street", "what road am i on",
            "où suis-je", "où sommes-nous", "quelle est ma position",
            "ma position", "ma localisation",
        ]
        _LOCATION_ARABIC = [
            "وين أنا", "فين أنا", "موقعي", "أين أنا",
            "وين موقعي", "فين موقعي", "ما موقعي",
        ]
        _is_location_query = (
            any(kw in _t_lower for kw in _LOCATION_LATIN)
            or any(kw in _t_ar   for kw in _LOCATION_ARABIC)
        )
        if _is_location_query:
            return {
                "intent": "navigate", "entities": [],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── C. NAVIGATE — go home (very common, always deterministic) ──
        _HOME_LATIN = [
            "go home", "take me home", "navigate home", "directions home",
            "take me to home", "bring me home", "i want to go home",
            "retour à la maison", "retourne à la maison", "rentrer à la maison",
            "emmène-moi à la maison", "amène-moi à la maison",
            "emmène-moi chez moi", "amène-moi chez moi",
        ]
        _HOME_ARABIC = [
            "ارجع للبيت", "ارجع لبيتي", "روح البيت", "روحني البيت",
            "روحني للبيت", "روحني لبيتي",
            "اريد العودة الى المنزل", "اريد العودة للمنزل",
            "عودة الى المنزل", "عودة للمنزل",
            "خذني الى المنزل", "خذني للمنزل", "خذني للبيت",
            "نحب نرجع للدار", "نحب نروح للدار", "نحب نروح لبيتي",
            "وديني للدار", "وديني للبيت",
        ]
        _is_home = (
            any(kw in _t_lower for kw in _HOME_LATIN)
            or any(kw in _t_ar   for kw in _HOME_ARABIC)
        )
        if _is_home:
            _home_target = {"en": "home", "fr": "maison",
                            "ar": "المنزل", "tn": "الدار"}.get(_lang_guess, "home")
            _clar_q = _CONFIRM_NAV_SPECIFIC.get(_lang_guess, _CONFIRM_NAV_SPECIFIC["en"]).format(entity=_home_target)
            self.pending_action = {
                "intent": "navigate", "entities": [_home_target],
                "mode": "NAVIGATE", "language": _lang_guess,
            }
            return {
                "intent": "navigate", "entities": [_home_target],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": _clar_q,
                "awaiting_confirmation": True, "confirmed": False,
                "parse_error": False, "is_specific_destination": True,
                "target": _home_target,
            }

        # ── C2. NAVIGATE TO SAVED LOCATION — "take me to work" ────────
        # Check if the utterance contains a saved named location label.
        # If found → navigate directly, no confirmation needed.
        try:
            from memory_service import get_named_location as _get_loc
            _NAV_TRIGGERS = [
                "take me to ", "take me at ", "navigate to ", "go to ",
                "bring me to ", "bring me at ", "i want to go to ",
                "emmène-moi à ", "amène-moi à ", "aller à ",
                "روحني لـ", "وديني لـ", "خذني لـ", "خذني إلى ",
            ]
            for _tr in _NAV_TRIGGERS:
                if _tr in _t_lower or _tr in _t_ar:
                    _src = _t_lower if _tr in _t_lower else _t_ar
                    _dest = _src.split(_tr, 1)[-1].strip(" ?.,!")
                    if _dest and _get_loc(_dest):
                        # It's a saved named location — navigate directly
                        self.pending_action = {
                            "intent": "navigate", "entities": [_dest],
                            "mode": "NAVIGATE", "language": _lang_guess,
                        }
                        _clar_q = _CONFIRM_NAV_SPECIFIC.get(
                            _lang_guess, _CONFIRM_NAV_SPECIFIC["en"]
                        ).format(entity=_dest)
                        return {
                            "intent": "navigate", "entities": [_dest],
                            "language": _lang_guess, "confidence": 0.97,
                            "source": "local_kw", "transcription": text,
                            "needs_clarification": False,
                            "clarification_question": _clar_q,
                            "awaiting_confirmation": True, "confirmed": False,
                            "parse_error": False, "is_specific_destination": True,
                            "target": _dest,
                        }
        except Exception:
            pass

        # ── D. MEMORY QUERY — "where did I last see my X?" ────────────
        # Must come BEFORE find-phone so "where did I find my phone last time"
        # goes to memory, not to the camera.
        _MEMORY_LATIN = [
            "where did i last see", "where did i see", "when did i last see",
            "where did i find", "where have i put",
            "where did i put", "where did i leave", "i lost my",
            "last time i saw", "last time i found",
            "où est mon", "où est ma", "où sont mes",
            "où ai-je mis", "où ai-je laissé", "quand ai-je vu",
            "où ai-je trouvé",
        ]
        _MEMORY_ARABIC = [
            "وين شفت", "فين شفت", "وين حطيت", "فين حطيت",
            "وين خليت", "فين خليت", "متى شفت", "آخر مرة شفت",
            "وين كان", "فين كان",
        ]
        _is_memory = (
            any(kw in _t_lower for kw in _MEMORY_LATIN)
            or any(kw in _t_ar   for kw in _MEMORY_ARABIC)
        )
        if _is_memory:
            _mem_target = ""
            for kw in sorted(_MEMORY_LATIN, key=len, reverse=True):
                if kw in _t_lower:
                    _after = _t_lower.split(kw, 1)[-1].strip()
                    _after = re.sub(r"^(my|your|the|mon|ma|mes|le|la|les)\s+", "", _after)
                    _mem_target = _after.strip(" ?.,!") or ""
                    break
            if not _mem_target:
                for kw in sorted(_MEMORY_ARABIC, key=len, reverse=True):
                    if kw in _t_ar:
                        _after = _t_ar.split(kw, 1)[-1].strip()
                        _mem_target = _after.strip(" ?.,!") or ""
                        break
            if not _mem_target:
                _mem_target = _t_lower.strip(" ?.,!")
            return {
                "intent": "memory_query", "entities": [_mem_target],
                "language": _lang_guess, "confidence": 0.95,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── E. NAME QUERY — "what is my name?" ───────────────────────
        _NAME_QUERY_LATIN = [
            "what is my name", "what's my name", "do you know my name",
            "who am i", "tell me my name", "say my name",
            "quel est mon nom", "comment je m'appelle", "tu connais mon nom",
        ]
        _NAME_QUERY_ARABIC = [
            "شنو اسمي", "ما اسمي", "قولي اسمي", "تعرف اسمي",
            "ما هو اسمي", "من أنا",
        ]
        _is_name_query = (
            any(kw in _t_lower for kw in _NAME_QUERY_LATIN)
            or any(kw in _t_ar   for kw in _NAME_QUERY_ARABIC)
        )
        if _is_name_query:
            return {
                "intent": "name_query", "entities": [],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False, "clarification_question": "",
                "awaiting_confirmation": False, "confirmed": True,
                "parse_error": False,
            }

        # ── F. FIND PHONE (most common find command) ───────────────────
        _PHONE_LATIN = [
            "find my phone", "find phone",
            "trouve mon téléphone",
        ]
        _PHONE_ARABIC = [
            "لقيلي تلفوني", "لقّيلي تلفوني",
            "ابحث عن هاتفي", "ابحث عن تلفوني",
        ]
        _is_find_phone = (
            any(kw in _t_lower for kw in _PHONE_LATIN)
            or any(kw in _t_ar   for kw in _PHONE_ARABIC)
        )
        if _is_find_phone:
            _ph = {"en": "phone", "fr": "téléphone",
                   "ar": "هاتف", "tn": "تلفون"}.get(_lang_guess, "phone")
            _clar_q = _CONFIRM_FIND.get(_lang_guess, _CONFIRM_FIND["en"]).format(entity=_ph)
            self.pending_action = {
                "intent": "env_action", "entities": [_ph],
                "mode": "FIND_OBJECT", "language": _lang_guess,
            }
            return {
                "intent": "env_action", "entities": [_ph],
                "language": _lang_guess, "confidence": 0.97,
                "source": "local_kw", "transcription": text,
                "needs_clarification": False,
                "clarification_question": _clar_q,
                "awaiting_confirmation": True, "confirmed": False,
                "parse_error": False, "sub_intent": "find", "target": _ph,
            }

        # ── 2. LLM call for everything else ───────────────────────────
        parsed = self._call_llm_intent(text)
        if not parsed:
            return self._fallback(text, reason="LLM unavailable")

        intent  = parsed.get("intent", "other")
        target  = (parsed.get("target") or "").strip()
        lang    = parsed.get("language", "en")
        is_spec = bool(parsed.get("is_specific_destination", False))
        conf    = float(parsed.get("confidence", 0.0))

        # Map LLM intents to internal intents
        internal_intent = {
            "find":     "env_action",
            "describe": "env_action",
            "navigate": "navigate",
            "other":    "unknown",
        }.get(intent, "unknown")

        if internal_intent == "unknown" or conf < 0.5:
            return {
                "intent":                 "unknown",
                "entities":               [],
                "language":               lang,
                "confidence":             conf,
                "source":                 "llm",
                "transcription":          text,
                "needs_clarification":    True,
                "clarification_question": _DID_NOT_UNDERSTAND.get(
                    lang, _DID_NOT_UNDERSTAND["en"]),
                "awaiting_confirmation":  False,
                "confirmed":              False,
                "parse_error":            False,
            }

        entities = [target] if target else []

        # ── 3. Build a localized confirmation question ──
        clar_q       = ""
        awaiting_cfm = False
        pending      = None

        if intent == "find" and target:
            clar_q = _CONFIRM_FIND.get(lang, _CONFIRM_FIND["en"]).format(entity=target)
            awaiting_cfm = True
            pending = {"intent": "env_action", "entities": entities,
                       "mode": "FIND_OBJECT", "language": lang}

        elif intent == "describe":
            clar_q = _CONFIRM_DESCRIBE.get(lang, _CONFIRM_DESCRIBE["en"])
            awaiting_cfm = True
            pending = {"intent": "env_action", "entities": entities,
                       "mode": "DESCRIBE", "language": lang}

        elif intent == "navigate" and target:
            # Check if target is a saved named location — treat as specific
            _is_saved_loc = False
            try:
                from memory_service import get_named_location as _get_loc
                if _get_loc(target):
                    _is_saved_loc = True
            except Exception:
                pass

            generic_by_hint = target.lower().strip() in _GENERIC_POI_HINTS
            specific = (_is_saved_loc) or (is_spec and not generic_by_hint)
            if specific:
                clar_q = _CONFIRM_NAV_SPECIFIC.get(
                    lang, _CONFIRM_NAV_SPECIFIC["en"]).format(entity=target)
            else:
                clar_q = _CONFIRM_NAV_GENERIC.get(
                    lang, _CONFIRM_NAV_GENERIC["en"]).format(entity=target)
            awaiting_cfm = True
            pending = {"intent": "navigate", "entities": entities,
                       "mode": "NAVIGATE", "language": lang}

        if pending:
            self.pending_action = pending

        # Track context for future LLM calls
        self._context.append({"text": text, "intent": internal_intent})

        return {
            "intent":                 internal_intent,
            "entities":               entities,
            "language":               lang,
            "confidence":             conf,
            "source":                 "llm",
            "transcription":          text,
            "needs_clarification":    False,
            "clarification_question": clar_q,
            "awaiting_confirmation":  awaiting_cfm,
            "confirmed":              False,
            "parse_error":            False,
        }

    # ── Confirmation handler ──────────────────────────────────

    def _handle_confirmation_reply(self, text: str) -> dict:
        action       = self.pending_action
        pending_lang = action.get("language", "en")
        answer       = _is_confirmation(text)

        if answer is True:
            self.pending_action = None
            return {
                "intent":                 action["intent"],
                "entities":               action["entities"],
                "language":               pending_lang,
                "confidence":             1.0,
                "source":                 "confirmation",
                "transcription":          text,
                "needs_clarification":    False,
                "clarification_question": "",
                "awaiting_confirmation":  False,
                "confirmed":              True,
                "parse_error":            False,
                "_force_replace":         action.get("_force_replace", False),
            }

        if answer is False:
            self.pending_action = None
            return {
                "intent":                 "unknown",
                "entities":               [],
                "language":               pending_lang,
                "confidence":             1.0,
                "source":                 "confirmation",
                "transcription":          text,
                "needs_clarification":    False,
                "clarification_question": _CANCELLED.get(
                    pending_lang, _CANCELLED["en"]),
                "awaiting_confirmation":  False,
                "confirmed":              False,
                "parse_error":            False,
            }

        # Neither YES nor NO — for navigate, treat as a specific
        # destination name; for everything else, re-prompt.
        if action["intent"] == "navigate":
            self.pending_action = None
            entities = ([text.strip()]
                        if "closest" not in text.lower()
                        and "nearest" not in text.lower()
                        else action["entities"])
            return {
                "intent":                 "navigate",
                "entities":               entities,
                "language":               pending_lang,
                "confidence":             0.9,
                "source":                 "confirmation_specific",
                "transcription":          text,
                "needs_clarification":    False,
                "clarification_question": "",
                "awaiting_confirmation":  False,
                "confirmed":              True,
                "parse_error":            False,
            }

        # Re-prompt for find/describe — keep pending alive
        return {
            "intent":                 "unknown",
            "entities":               [],
            "language":               pending_lang,
            "confidence":             0.0,
            "source":                 "confirmation_reprompt",
            "transcription":          text,
            "needs_clarification":    False,
            "clarification_question": _REPROMPT_YES_NO.get(
                pending_lang, _REPROMPT_YES_NO["en"]),
            "awaiting_confirmation":  True,
            "confirmed":              False,
            "parse_error":            False,
        }

    # ── Whisper ───────────────────────────────────────────────

    def _transcribe(self, wav_path: str) -> str:
        if not self.whisper:
            print("[STT] Whisper not loaded")
            return ""
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 200:
            return ""
        try:
            segments, info = self.whisper.transcribe(
                wav_path, beam_size=5, vad_filter=True,
                no_speech_threshold=0.5, language=None)
            text = " ".join(s.text for s in segments).strip()
            print(f"[STT] lang={info.language} p={info.language_probability:.2f} → \"{text}\"")
            # Map Whisper's language code to our 4-language scheme.
            # Whisper returns 'ar' for any Arabic — we can't tell MSA from
            # Tunisian Derja from STT alone.  Default to 'tn' since this app
            # primarily serves Tunisian users; the LLM will refine to 'ar'
            # if the utterance is in MSA style.
            stt_lang = (info.language or "en").lower()
            self._last_stt_lang = {
                "ar": "tn",   # default Arabic to Tunisian (most common here)
                "fr": "fr",
                "en": "en",
            }.get(stt_lang, "en")
            return text
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return ""

    # ── LLM intent call ───────────────────────────────────────

    def _call_llm_intent(self, text: str) -> Optional[dict]:
        # Priority: OpenRouter → Groq (fast, free) → Gemini (reliable backup)
        if not self.api_key and not self.google_api_key and not self._groq_key:
            print("[LLM] No API keys configured")
            return None

        msgs = [{"role": "system", "content": _INTENT_SYSTEM_PROMPT}]
        for utt, jsn in _FEW_SHOT_EXAMPLES:
            msgs.append({"role": "user",     "content": utt})
            msgs.append({"role": "assistant", "content": jsn})
        msgs.append({"role": "user", "content": text})

        if self.api_key:
            parsed = self._call_openrouter_intent(msgs)
            if parsed:
                return parsed

        if self._groq_key:
            parsed = self._call_groq_intent(text)
            if parsed:
                return parsed

        if self.google_api_key:
            print("[LLM] Trying Gemini…")
            parsed = self._call_gemini_intent(msgs, text)
            if parsed:
                return parsed

        return None

    def _call_openrouter_intent(self, msgs: list) -> Optional[dict]:
        models = [self.llm_model] + list(self.llm_fallbacks)
        models = [m for m in models
                  if m not in self._dead_models
                  and m not in self._always_skip]

        for mi, model in enumerate(models):
            try:
                r = requests.post(
                    self.llm_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type":  "application/json"},
                    json={"model":       model,
                          "messages":    msgs,
                          "max_tokens":  120,
                          "temperature": 0.0,
                          "response_format": {"type": "json_object"}},
                    timeout=10)
                if r.status_code == 200:
                    body    = r.json() or {}
                    choices = body.get("choices") or []
                    if not choices:
                        print(f"[LLM] {model} no choices — skip")
                        continue
                    raw = ((choices[0] or {}).get("message") or {}).get("content") or ""
                    if not raw.strip():
                        print(f"[LLM] {model} empty content — blacklisting")
                        self._always_skip.add(model)
                        continue
                    parsed = self._parse_json(raw)
                    if parsed:
                        if mi > 0:
                            print(f"[LLM] ok via OpenRouter #{mi}: {model}")
                        return parsed
                    print(f"[LLM] {model} non-JSON: {raw[:80]}")
                    continue
                print(f"[LLM] {model} → {r.status_code}: {r.text[:100]}")
                if r.status_code == 404:
                    self._dead_models.add(model)
                elif r.status_code == 400:
                    self._always_skip.add(model)
                # 429 = rate limit — try next, don't blacklist
            except requests.Timeout:
                print(f"[LLM] {model} timed out")
            except Exception as e:
                print(f"[LLM] {model} error: {e}")
        return None

    def _call_groq_intent(self, text: str) -> Optional[dict]:
        """Groq Llama-4-Scout: ~0.5s, free, ~500 RPD. Text-only."""
        prompt = (
            _INTENT_SYSTEM_PROMPT
            + "\n\nOutput ONLY the JSON object.\n\nUser: " + text
        )
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._groq_key}",
                         "Content-Type":  "application/json"},
                json={"model":           "meta-llama/llama-4-scout-17b-16e-instruct",
                      "max_tokens":      150,
                      "temperature":     0.0,
                      "response_format": {"type": "json_object"},
                      "messages":        [{"role": "user", "content": prompt}]},
                timeout=8,
            )
            if r.status_code != 200:
                print(f"[LLM groq] {r.status_code}: {r.text[:80]}")
                return None
            raw    = r.json()["choices"][0]["message"].get("content", "")
            parsed = self._parse_json(raw)
            if parsed:
                print("[LLM] ok via Groq (Llama-4-Scout)")
                return parsed
        except requests.Timeout:
            print("[LLM groq] timed out")
        except Exception as e:
            print(f"[LLM groq] error: {e}")
        return None

    def _call_gemini_intent(self, msgs: list, user_text: str) -> Optional[dict]:
        if not self.google_api_key:
            return None
        sys_text   = next((m["content"] for m in msgs if m["role"] == "system"), "")
        contents: list = []
        first_user = True
        for m in msgs:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            txt  = m["content"]
            if first_user and role == "user":
                txt        = sys_text + "\n\n" + txt
                first_user = False
            contents.append({"role": role, "parts": [{"text": txt}]})
        url = (f"{self.gemini_url}/gemini-2.5-flash-lite:generateContent"
               f"?key={self.google_api_key}")
        try:
            r = requests.post(url,
                              headers={"Content-Type": "application/json"},
                              json={"contents": contents,
                                    "generationConfig": {
                                        "temperature":      0.0,
                                        "maxOutputTokens":  200,
                                        "responseMimeType": "application/json",
                                    }},
                              timeout=20)   # was 12 — caused timeouts
            if r.status_code != 200:
                print(f"[LLM gemini] {r.status_code}: {r.text[:150]}")
                return None
            cands = (r.json() or {}).get("candidates") or []
            if not cands:
                return None
            raw = "".join(p.get("text", "")
                          for p in (cands[0].get("content") or {}).get("parts") or [])
            parsed = self._parse_json(raw)
            if parsed:
                print("[LLM] ok via Gemini")
                return parsed
        except requests.Timeout:
            print("[LLM gemini] timed out (20s)")
        except Exception as e:
            print(f"[LLM gemini] error: {e}")
        return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Parse the LLM's JSON, tolerating markdown fences and
        leading/trailing prose."""
        if not raw:
            return None
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        # Find the JSON object
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    # ── Fallback / unknown response ───────────────────────────

    def _fallback(self, text: str, reason: str = "") -> dict:
        if reason:
            print(f"[DialogueAgent] fallback: {reason}")
        # Use Whisper's language hint so we apologize in the user's
        # language, not in English.
        lang = self._last_stt_lang or "en"
        return {
            "intent":                 "unknown",
            "entities":               [],
            "language":               lang,
            "confidence":             0.0,
            "source":                 "fallback",
            "transcription":          text,
            "needs_clarification":    True,
            "clarification_question": _DID_NOT_UNDERSTAND.get(
                lang, _DID_NOT_UNDERSTAND["en"]),
            "awaiting_confirmation":  False,
            "confirmed":              False,
            "parse_error":            False,
        }

    # ── Pending-action management ─────────────────────────────

    def reset_pending(self):
        """Clear pending action and short conversational context.
        Called when the user closes a screen or the action completes."""
        self.pending_action = None
        self._context.clear()

    # Backwards-compatible alias for any caller that still uses the
    # old method name from v5 (main.py /reset endpoint).
    def reset_context(self):
        self.reset_pending()