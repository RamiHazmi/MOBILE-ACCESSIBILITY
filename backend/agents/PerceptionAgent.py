"""
PerceptionAgent.py — v4 (free models + temporal consensus)
─────────────────────────────────────────────────────────────
What changed vs v3:

1. SWITCHED TO FREE MODEL
   gemini-1.5-flash on OpenRouter is NOT free.
   Now uses google/gemma-3-27b-it:free with a fallback chain to
   nvidia/nemotron-nano-12b-v2-vl:free.

2. STRONGER FIND-OBJECT PROMPT
   The prompt now emphasises "common household items at any angle,
   even partially visible, including reflections and screens-off",
   which fixes the dark-phone-on-desk failure (image 2).

3. TEMPORAL CONSENSUS  (key fix for "I cannot see your phone")
   Single-shot detection misses on shaky/blurry frames.  We now
   keep a rolling window of the last N frames; the target is only
   declared "found" after it has been positively detected in
   VISION_MIN_CONFIRMATIONS frames (default: 2 of last 3).
   Conversely the "I cannot see it" message only fires after 3
   consecutive negative scans, so transient blur doesn't trigger
   premature "not found" speech.

4. BOUNDING-BOX PROMPT IS EXPLICIT
   The model is instructed to OUTPUT bbox_cx/cy/w/h ONLY for the
   target if found, plus for hazards.  No more giant flood of
   boxes from background clutter.  Boxes only render WHEN found.

5. RETRY + FALLBACK MODEL
   If the primary model fails or is rate-limited, we automatically
   retry with the secondary :free model.  Both are zero-cost.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections import deque
from typing import Optional

import cv2
import requests


# ─── Synonym table (target detection robustness) ────────────────

_SYNONYMS: dict[str, frozenset[str]] = {
    "laptop":   frozenset({"laptop", "notebook", "macbook", "computer",
                           "chromebook", "pc", "ordinateur"}),
    "phone":    frozenset({"phone", "smartphone", "mobile", "cellphone",
                           "iphone", "android", "cell phone", "telephone",
                           "téléphone", "هاتف", "جوال"}),
    "glasses":  frozenset({"glasses", "spectacles", "eyeglasses",
                           "sunglasses", "lunettes", "نظارات"}),
    "keys":     frozenset({"keys", "key", "keychain", "clé", "clés", "مفاتيح"}),
    "wallet":   frozenset({"wallet", "purse", "billfold", "portefeuille",
                           "محفظة"}),
    "remote":   frozenset({"remote", "controller", "tv remote",
                           "télécommande"}),
    "book":     frozenset({"book", "notebook", "textbook", "livre", "كتاب"}),
    "cup":      frozenset({"cup", "mug", "glass", "tasse", "كأس", "كوب"}),
    "bottle":   frozenset({"bottle", "water bottle", "bouteille", "قارورة"}),
    "chair":    frozenset({"chair", "seat", "stool", "wheelchair", "chaise",
                           "كرسي"}),
    "door":     frozenset({"door", "doorway", "exit", "entrance", "porte",
                           "باب"}),
    "table":    frozenset({"table", "desk", "counter", "bureau", "طاولة"}),
    "bag":      frozenset({"bag", "backpack", "purse", "handbag", "suitcase",
                           "sac", "حقيبة"}),
    "charger":  frozenset({"charger", "cable", "cord", "wire", "chargeur",
                           "câble", "شاحن", "كابل"}),
}

_HAZARD_LABELS: frozenset[str] = frozenset({
    "stairs", "step", "steps", "curb", "drop", "hole",
    "low ceiling", "glass door", "gate", "ledge", "ramp",
    "pothole", "open manhole", "broken glass", "wet floor",
    "edge", "stairway", "staircase",
})

# Labels that should NEVER trigger HIGH danger just because they're
# close+center.  A person sitting on a sofa right in front of you is
# not a hazard, it's a normal living-room scene.  Same for furniture,
# household items, and pets.  These can still be flagged as HIGH if
# they are explicitly "moving toward user" (e.g. a moving car).
_NEVER_HAZARD_LABELS: frozenset[str] = frozenset({
    # People
    "person", "people", "man", "woman", "child", "kid", "boy", "girl",
    "human", "face",
    # Furniture
    "sofa", "couch", "chair", "armchair", "stool", "bench", "seat",
    "bed", "table", "desk", "shelf", "bookshelf", "cabinet", "wardrobe",
    "drawer", "nightstand", "tv stand", "coffee table", "dining table",
    # Soft / household
    "cushion", "pillow", "blanket", "rug", "carpet", "curtain", "lamp",
    "tv", "television", "screen", "monitor", "laptop", "computer",
    "phone", "remote", "book", "magazine", "plant", "vase", "picture",
    "painting", "mirror", "clock", "speaker",
    # Kitchen
    "cup", "mug", "glass", "bottle", "plate", "bowl", "fork", "spoon",
    "knife", "pot", "pan",
    # Pets
    "cat", "dog", "pet",
})


def _repair_truncated_json(s: str) -> str:
    """
    Best-effort repair for VLM JSON responses that get cut off at
    max_tokens.  Trims to the last complete value and closes any
    open arrays / objects.
    """
    # Trim any trailing partial token after the last complete delimiter
    last_safe = max(s.rfind('}'), s.rfind(']'),
                    s.rfind('"'), s.rfind(','))
    if last_safe > 0:
        s = s[:last_safe + 1]
    # Drop trailing comma if the truncation left one dangling
    s = re.sub(r',\s*$', '', s)
    # Count unclosed brackets / braces and append closers
    open_braces  = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')
    s += ']' * max(0, open_brackets)
    s += '}' * max(0, open_braces)
    return s


def _parse_prose(text: str, find_target: Optional[str]) -> Optional[dict]:
    """
    Last-ditch parser for when the VLM ignores the JSON instruction
    and replies in prose like "A phone is in the center, close by."

    If we have a find_target and the prose mentions it, build a minimal
    scene with target_found=True so the user gets useful feedback
    instead of "camera unclear".

    Returns None if even prose extraction fails.
    """
    if not text:
        return None
    text_lc = text.lower()

    # Detect position from prose
    pos = "center"
    if any(w in text_lc for w in ["left", "à gauche", "اليسار", "يسار"]):
        pos = "left"
    elif any(w in text_lc for w in ["right", "à droite", "اليمين", "يمين"]):
        pos = "right"

    # Detect distance
    dist = "nearby"
    if any(w in text_lc for w in ["close", "near", "in front", "right there",
                                    "près", "devant", "قريب", "أمامك"]):
        dist = "close"
    elif any(w in text_lc for w in ["far", "loin", "بعيد"]):
        dist = "far"

    # If we have a find_target and the prose mentions it, count it as found
    if find_target:
        target_lc = find_target.lower()
        # Match the target word OR any of its English/Arabic synonyms
        target_synonyms = {target_lc}
        for canonical, syns in _SYNONYMS.items():
            if target_lc in syns or target_lc == canonical:
                target_synonyms.update(syns)
                target_synonyms.add(canonical)
        if any(t in text_lc for t in target_synonyms):
            scene = _empty_scene(find_target)
            scene["target_found"]    = True
            scene["target_position"] = pos
            scene["target_distance"] = dist
            scene["confidence"]      = "MEDIUM"   # prose is less reliable
            scene["objects"]         = [{
                "label":    find_target,
                "position": pos,
                "distance": dist,
                "threat":   False,
                "moving":   False,
                "size":     "medium",
            }]
            _g = text.strip()[:140]
            scene["guidance"] = "" if _g.lstrip().startswith(('{', '[')) else _g
            scene["parse_error"] = False
            scene["consensus_state"] = "FOUND"  # let the orchestrator speak
            return scene

    # No find_target match — return a generic scene with the prose as guidance
    # so the danger_node has *something* localizable to say.
    scene = _empty_scene(find_target)
    scene["confidence"] = "MEDIUM"
    _g = text.strip()[:140]
    scene["guidance"]   = "" if _g.lstrip().startswith(('{', '[')) else _g
    scene["parse_error"] = False
    return scene


def _matches_target(label: str, target: str) -> bool:
    label_lc  = label.lower().strip()
    target_lc = target.lower().strip()
    if not target_lc:
        return False
    if target_lc in label_lc or label_lc in target_lc:
        return True
    lw = set(label_lc.split())
    tw = set(target_lc.split())
    if lw & tw:
        return True
    synonyms = _SYNONYMS.get(target_lc, frozenset({target_lc}))
    if lw & synonyms:
        return True
    # Cross-check: maybe the label is a synonym of something the
    # target maps to
    for canonical, syns in _SYNONYMS.items():
        if target_lc in syns and (canonical in label_lc or label_lc in syns):
            return True
    return False


def _empty_scene(find_target: Optional[str] = None) -> dict:
    base = {
        "environment":        "unknown",
        "scene_type":         "other",
        "danger_level":       "LOW",
        "is_dangerous":       False,
        "danger_message":     "",
        "objects":            [],
        "immediate_threats":  [],
        "safe_path":          "center",
        "navigation_command": "GO_FORWARD",
        "guidance":           "",
        "confidence":         "LOW",
        "parse_error":        True,
    }
    if find_target:
        base.update({"target_found": False, "target_position": None,
                     "target_distance": None, "target_bbox": None})
    return base


# ─── Perception Agent ───────────────────────────────────────────
from config import (
    OPENROUTER_API_KEY,
    VLM_MODEL_VISION,
    VLM_MODEL_FALLBACK,
    VLM_MODEL_EXTRA_FALLBACKS,
    VISION_MIN_CONFIRMATIONS,
    VISION_MAX_HISTORY
)
class PerceptionAgent:
    """
    Vision agent for blind users.  Uses ONLY free OpenRouter models.
    Implements temporal consensus to avoid single-frame misses.
    """

    _SYSTEM_PROMPT = """\
You are the VISION MODULE for smart glasses worn by a totally blind person.
Errors can cause physical harm. Be accurate. Never invent objects.

DISTANCE (use exactly these words):
  close  = under 1.5 m
  medium = 1.5 – 4 m
  far    = over 4 m

POSITION (from wearer's view):  left | center | right

BOUNDING BOX (normalised 0.0–1.0, origin top-left):
  bbox_cx = horizontal center  (0=left, 1=right)
  bbox_cy = vertical center    (0=top, 1=bottom)
  bbox_w  = width fraction
  bbox_h  = height fraction

DANGER RULES (apply in order, stop at first match):
  HIGH   → close+center, OR moving toward user, OR label contains
            stairs/step/curb/drop/hole/ledge/ramp/gate/broken glass
  MEDIUM → close+left or close+right
  LOW    → otherwise

If unsure of an object's identity, label it "unknown object"."""

    def __init__(
        self,
        api_key: str,
        source = 0,
        mode: str = "external",
        video_path: Optional[str] = None,
        primary_model: str = VLM_MODEL_VISION,
        fallback_model: str = VLM_MODEL_FALLBACK,
        extra_fallbacks: Optional[list] = VLM_MODEL_EXTRA_FALLBACKS,
        google_api_key: Optional[str] = None,
        min_confirmations: int = VISION_MIN_CONFIRMATIONS,
        history_size:      int = VISION_MAX_HISTORY,
    ):
        self.api_key        = api_key
        self.google_api_key = (google_api_key or "").strip()
        self.url            = "https://openrouter.ai/api/v1/chat/completions"
        self.gemini_url     = "https://generativelanguage.googleapis.com/v1beta/models"
        self.mode     = mode
        self.primary  = primary_model
        self.fallback = fallback_model
        # Build the full rotation list — primary first, then explicit
        # fallback, then any extra fallbacks passed in from config.
        # Models prefixed with "gemini/" route to Google AI Studio
        # (1000 RPD on Flash-Lite, much more generous than OpenRouter).
        # Skipped if no Google API key is configured.
        _seen: set = set()
        _chain: list = []
        for m in [primary_model, fallback_model] + (extra_fallbacks or []):
            if not m or m in _seen:
                continue
            _seen.add(m)
            if m.startswith("gemini/") and not self.google_api_key:
                print(f"[Perception] Skipping {m} — GOOGLE_API_KEY not set")
                continue
            if not m.startswith("gemini/") and not self.api_key:
                print(f"[Perception] Skipping {m} — OPENROUTER_API_KEY not set")
                continue
            _chain.append(m)
        self._models = _chain
        if self._models:
            print(f"[Perception] Model rotation: {self._models}")
        else:
            print("[Perception] WARNING: no models configured — set "
                  "GOOGLE_API_KEY and/or OPENROUTER_API_KEY")
        # Rate-limit memory: model_name → unix_ts when we can try again
        self._cooloff_until: dict = {}
        # Models that have permanently 404'd this session
        self._dead: set = set()

        if mode == "cam":
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
        elif mode == "video":
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open video")
        else:
            self.cap = None  # external mode — frames arrive over HTTP

        self.frame_skip        = 5
        self.request_delay     = 1.5
        self.last_request_time = 0.0
        self.frame_count       = 0
        self.width             = 256
        self.height            = 192
        self.last_result: Optional[dict] = None

        # Temporal consensus state — keyed by find_target
        self._history:  dict[str, deque] = {}
        self._min_conf  = min_confirmations
        self._hist_size = history_size

    # ── Encode ────────────────────────────────────────────────

    def encode(self, frame) -> str:
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buf).decode("utf-8")

    # ── Prompt builder ────────────────────────────────────────

    def build_prompt(self, find_target: Optional[str] = None) -> str:
        obj_schema = """\
  {
    "label":              "exact short noun, or 'unknown object'",
    "position":           "left" | "center" | "right",
    "distance":           "close" | "medium" | "far",
    "moving":             true | false,
    "moving_toward_user": true | false,
    "threat":             true | false,
    "bbox_cx": 0.0,
    "bbox_cy": 0.0,
    "bbox_w":  0.0,
    "bbox_h":  0.0
  }"""

        schema = f"""\
Respond with EXACTLY this JSON, no extra keys, no markdown:
{{
  "environment": "indoor" | "outdoor" | "unknown",
  "scene_type":  "street" | "corridor" | "room" | "crosswalk" | "stairs" | "shop" | "other",
  "confidence":  "HIGH" | "MEDIUM" | "LOW",
  "danger_level": "LOW" | "MEDIUM" | "HIGH",
  "objects": [
{obj_schema}
  ],
  "immediate_threats":   ["short threat phrase"],
  "safe_path":           "left" | "center" | "right" | "none",
  "navigation_command":  "GO_FORWARD" | "MOVE_LEFT" | "MOVE_RIGHT" | "STOP" | "WAIT",
  "guidance": "one calm sentence, max 12 words"
}}"""

        if find_target:
            return f"""\
=== SEARCH TASK (TOP PRIORITY) ===
The blind user is searching for: "{find_target}"

DETECTION RULES:
  • Look at EVERY region of the image — corners, edges, on top of
    furniture, on the floor, on the desk surface.
  • The "{find_target}" may be:
      - powered off (a dark rectangle, e.g. an unlit phone screen)
      - partially visible (cable plugged in, edge sticking out)
      - lying flat on a surface, not standing up
      - reflective, dark, or low-contrast against background
  • Synonyms accepted: phone=mobile=smartphone=cellphone=téléphone,
    laptop=notebook=computer, keys=keychain, glasses=spectacles,
    bag=backpack=purse, charger=cable=cord.
  • DO NOT confuse the target with similar-shaped furniture.

ADD these fields to the root JSON object:
  "target_found":    true | false,
  "target_position": "left" | "center" | "right" | null,
  "target_distance": "close" | "medium" | "far" | null,
  "target_bbox":     {{"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}} | null

INSTRUCTIONS for the objects[] array:
  • Always list the target if you see it, with its bbox.
  • List up to 5 obstacles/hazards relevant for safe navigation.
  • DO NOT list every random background object — keep it focused.

If target NOT found: "target_found": false, others null.

CONFIDENCE FIELD (critical — controls how fast the user is told):
  • "HIGH"   → you can clearly see the {find_target} and you are sure
              it is the {find_target} (not a similar-shaped object).
              Use HIGH whenever you are sure — don't be timid, the
              user is waiting and a single HIGH triggers an instant
              "found" announcement.
  • "MEDIUM" → you think it is the {find_target} but the lighting,
              angle, or partial view leaves ambiguity.
  • "LOW"    → you are guessing.
=== END SEARCH TASK ===

{schema}"""

        return ("Analyse the image for a blind user.  List up to 6 objects "
                "of navigational interest.\n\n" + schema)

    # ── VLM call (with fallback model) ────────────────────────

    def ask(
        self,
        img_b64: str,
        find_target: Optional[str] = None,
        max_retries: int = 1,
    ) -> Optional[str]:
        # Either OpenRouter or Google key is enough — we'll only
        # attempt the providers we have keys for (filtered at __init__).
        if not self.api_key and not self.google_api_key:
            print("[Perception] No API keys configured (need OPENROUTER_API_KEY "
                  "or GOOGLE_API_KEY)")
            return None

        now = time.time()

        # Models that are permanently dead this session (404 / endpoint gone)
        live_models = [m for m in self._models if m not in self._dead]
        if not live_models:
            print("[Perception] ALL models are permanently dead — cannot proceed")
            return None

        # Filter to models not currently in their rate-limit cooloff
        candidates = [m for m in live_models
                      if self._cooloff_until.get(m, 0) <= now]

        if not candidates:
            # All live models are cooling off.  If the soonest one is
            # >1h away, that's the daily-quota case — give up.
            soonest = min(live_models,
                          key=lambda m: self._cooloff_until.get(m, 0))
            wait = max(0, self._cooloff_until.get(soonest, 0) - now)
            if wait > 3600:
                print(f"[Perception] all models out of DAILY quota "
                      f"({int(wait/60)} min until reset) — giving up")
                return None
            print(f"[Perception] all models cooled off; waiting {min(wait, 5.0):.1f}s "
                  f"for {soonest}")
            time.sleep(min(wait, 5.0))
            candidates = [soonest]

        for model in candidates:
            is_gemini = model.startswith("gemini/")
            if is_gemini:
                gem_id = model[len("gemini/"):]
                ok, status, content = self._call_gemini(
                    gem_id, img_b64, find_target)
            else:
                ok, status, content = self._call_openrouter(
                    model, img_b64, find_target)

            if ok and content:
                return content

            # Failure — categorize and decide whether to retry/fallback
            if status == 404:
                self._dead.add(model)
                print(f"[Perception] {model} marked DEAD (404)")
                continue
            if status in (429, 503):
                lc = (content or "").lower()
                is_daily = ("free-models-per-day" in lc
                            or "per-day" in lc
                            or "resource_exhausted" in lc
                            or "quota" in lc)
                if is_daily:
                    self._cooloff_until[model] = time.time() + 86400.0
                    print(f"[Perception] {model} DAILY QUOTA EXHAUSTED — "
                          f"cooled off for 24h")
                else:
                    self._cooloff_until[model] = time.time() + 90.0
                    print(f"[Perception] {model} throttled, cooled off 90s")
                continue
            # Other status: try next model
            print(f"[Perception] {model} → status={status}, falling back")

        return None

    # ── Person identification ─────────────────────────────────

    def ask_person_id(self, img_b64: str,
                      persons: list) -> Optional[dict]:
        """
        Identify a saved person visible in the current frame.

        Uses IMAGE-TO-IMAGE comparison when saved image_b64 exists —
        far more accurate than description-only matching.

        persons: list of {name, relationship, face_description, image_b64}
        Returns: {match, relationship, confidence, position} or None.
        Only returns a match with confidence "high" to avoid false positives.
        """
        if not persons or not img_b64:
            return None
        if not self.api_key and not self.google_api_key:
            return None

        # Separate persons with saved photo vs description-only
        with_photo    = [p for p in persons[:5] if p.get("image_b64")]
        without_photo = [p for p in persons[:5] if not p.get("image_b64")]

        now  = time.time()
        live = [m for m in self._models
                if m not in self._dead
                and self._cooloff_until.get(m, 0) <= now]
        if not live:
            return None

        # ── Phase 1: image-to-image comparison (most reliable) ──
        if with_photo:
            result = self._person_id_with_images(
                img_b64, with_photo, live)
            if result and result.get("match"):
                return result
            # If no match from photo comparison, don't fall through
            # to description matching for the same persons (avoid doubles)
            without_photo = [p for p in without_photo
                             if p.get("name") not in
                             {x.get("name") for x in with_photo}]

        # ── Phase 2: description-only (only if no saved photos) ──
        if without_photo:
            result = self._person_id_with_desc(
                img_b64, without_photo, live)
            if result and result.get("match"):
                return result

        return {"match": None, "confidence": "low", "position": "center"}

    def _person_id_with_images(self, img_b64: str,
                               persons: list, live_models: list) -> Optional[dict]:
        """
        Face comparison by showing the current frame AND each saved face photo
        side-by-side in a single VLM request.
        """
        # Build per-person text labels
        name_list = ", ".join(
            f"{p['name']} ({p.get('relationship','?')})" for p in persons)

        intro = (
            "You are a face recognition assistant helping a blind user.\n\n"
            f"The FIRST image is the current camera frame.\n"
            f"The following images are saved reference faces: {name_list}.\n\n"
            "Compare the face(s) in the FIRST image against each reference.\n"
            "Requirements for a match:\n"
            "  • The face in the current frame must be CLEARLY VISIBLE (not blurry, not from behind).\n"
            "  • The facial features (skin tone, face shape, hair) must visibly match a reference.\n"
            "  • Do NOT guess — if unsure, return null.\n\n"
            'Return ONLY this JSON (no markdown):\n'
            '{"match": "<exact name>" | null, '
            '"confidence": "high" | "medium" | "low", '
            '"position": "left" | "center" | "right"}\n\n'
            "Only set match to a name when confidence is high."
        )

        for model in live_models[:2]:
            try:
                is_gemini = model.startswith("gemini/")
                if is_gemini:
                    parts: list = [{"text": intro}]
                    # Current frame
                    parts.append({"text": "CURRENT FRAME:"})
                    parts.append({"inline_data": {
                        "mime_type": "image/jpeg", "data": img_b64}})
                    # Reference faces
                    for p in persons:
                        parts.append({"text":
                            f"REFERENCE — {p['name']} ({p.get('relationship','?')}):"})
                        parts.append({"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": p["image_b64"]}})

                    gem_id = model[len("gemini/"):]
                    url = (f"{self.gemini_url}/{gem_id}:generateContent"
                           f"?key={self.google_api_key}")
                    payload = {
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "maxOutputTokens": 80,
                            "temperature":     0.0,
                        },
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    if r.status_code != 200:
                        print(f"[person_id/img] gemini {r.status_code}: {r.text[:100]}")
                        continue
                    raw = (r.json().get("candidates", [{}])[0]
                           .get("content", {})
                           .get("parts", [{}])[0]
                           .get("text", ""))
                else:
                    content: list = [{"type": "text", "text": intro}]
                    content.append({"type": "text", "text": "CURRENT FRAME:"})
                    content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"}})
                    for p in persons:
                        content.append({"type": "text",
                            "text": f"REFERENCE — {p['name']} ({p.get('relationship','?')}):"})
                        content.append({"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{p['image_b64']}"}})

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://blindglasses.local",
                        "X-Title":       "Blind Glasses Assistant",
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens":  80,
                        "temperature": 0.0,
                    }
                    r = requests.post(self.url, headers=headers,
                                      json=payload, timeout=15)
                    if r.status_code != 200:
                        print(f"[person_id/img] openrouter {r.status_code}: {r.text[:100]}")
                        continue
                    raw = r.json()["choices"][0]["message"]["content"]

                return self._parse_person_id_response(raw, persons)

            except Exception as e:
                print(f"[person_id/img] {model} error: {e}")
                continue
        return None

    def _person_id_with_desc(self, img_b64: str,
                             persons: list, live_models: list) -> Optional[dict]:
        """Description-only fallback when no saved photos exist."""
        lines = []
        for i, p in enumerate(persons, 1):
            name = (p.get("name") or "").strip()
            rel  = (p.get("relationship") or "person").strip()
            desc = (p.get("face_description") or "").strip()[:140]
            if not name or not desc:
                continue
            lines.append(f"{i}. {name} ({rel}): {desc}")
        if not lines:
            return None

        prompt = (
            "You are a face recognition assistant helping a blind user.\n\n"
            "Saved contacts with face descriptions:\n" + "\n".join(lines) + "\n\n"
            "Look at the person in this image.\n"
            "Requirements for a match:\n"
            "  • The face must be CLEARLY VISIBLE and facing the camera.\n"
            "  • The physical features described must match what you observe.\n"
            "  • Do NOT guess — only match if you are HIGHLY confident.\n\n"
            'Return ONLY this JSON (no markdown):\n'
            '{"match": "<exact name>" | null, '
            '"confidence": "high" | "medium" | "low", '
            '"position": "left" | "center" | "right"}\n\n'
            "If the face is unclear or features don't clearly match, return null."
        )

        for model in live_models[:2]:
            try:
                is_gemini = model.startswith("gemini/")
                if is_gemini:
                    gem_id = model[len("gemini/"):]
                    url = (f"{self.gemini_url}/{gem_id}:generateContent"
                           f"?key={self.google_api_key}")
                    payload = {
                        "contents": [{"parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg",
                                             "data": img_b64}},
                        ]}],
                        "generationConfig": {
                            "maxOutputTokens": 80, "temperature": 0.0},
                    }
                    r = requests.post(url, json=payload, timeout=12)
                    if r.status_code != 200:
                        continue
                    raw = (r.json().get("candidates", [{}])[0]
                           .get("content", {})
                           .get("parts", [{}])[0]
                           .get("text", ""))
                else:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://blindglasses.local",
                        "X-Title":       "Blind Glasses Assistant",
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"}},
                        ]}],
                        "max_tokens": 80, "temperature": 0.0,
                    }
                    r = requests.post(self.url, headers=headers,
                                      json=payload, timeout=12)
                    if r.status_code != 200:
                        continue
                    raw = r.json()["choices"][0]["message"]["content"]

                return self._parse_person_id_response(raw, persons)

            except Exception as e:
                print(f"[person_id/desc] {model} error: {e}")
                continue
        return None

    def describe_face(self, img_b64: str,
                      person_name: str = "",
                      relationship: str = "") -> Optional[str]:
        """
        Return a compact face description suitable for later recognition.
        Used when saving a new person so we can match them visually later.
        """
        if not img_b64:
            return None
        who = f"{person_name} ({relationship})" if person_name else "the person"
        prompt = (
            f"Describe the face of {who} in this image for future recognition.\n"
            "Include (if visible): skin tone, approximate age, hair color and style, "
            "eye color, face shape, beard/mustache, glasses, any distinctive features.\n"
            "Be specific and factual. Max 3 sentences. No JSON — just plain text."
        )

        now  = time.time()
        live = [m for m in self._models
                if m not in self._dead
                and self._cooloff_until.get(m, 0) <= now]
        if not live:
            return None

        for model in live[:2]:
            try:
                is_gemini = model.startswith("gemini/")
                if is_gemini:
                    gem_id = model[len("gemini/"):]
                    url = (f"{self.gemini_url}/{gem_id}:generateContent"
                           f"?key={self.google_api_key}")
                    payload = {
                        "contents": [{"parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg",
                                             "data": img_b64}},
                        ]}],
                        "generationConfig": {
                            "maxOutputTokens": 200,
                            "temperature":     0.1,
                        },
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    if r.status_code != 200:
                        continue
                    text = (r.json().get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")).strip()
                else:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://blindglasses.local",
                        "X-Title":       "Blind Glasses Assistant",
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"}},
                        ]}],
                        "max_tokens":  200,
                        "temperature": 0.1,
                    }
                    r = requests.post(self.url, headers=headers,
                                      json=payload, timeout=15)
                    if r.status_code != 200:
                        continue
                    text = r.json()["choices"][0]["message"]["content"].strip()

                if text and not text.lstrip().startswith("{"):
                    return text
            except Exception as e:
                print(f"[describe_face] {model} error: {e}")
                continue
        return None

    def _parse_person_id_response(self, raw: str,
                                  persons: list) -> Optional[dict]:
        """Parse VLM JSON response; only trust 'high' confidence matches."""
        try:
            raw = raw.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result     = json.loads(raw)
            match_name = result.get("match")
            confidence = result.get("confidence", "low")

            # Reject anything below high confidence — false positives are worse
            # than missed detections for a blind user.
            if not match_name or confidence not in ("high",):
                return {"match": None, "confidence": "low", "position": "center"}

            matched = next(
                (p for p in persons
                 if (p.get("name") or "").lower() == match_name.lower()),
                None,
            )
            rel = (matched.get("relationship") or "") if matched else ""
            print(f"[person_id] ✓ match={match_name} rel={rel} "
                  f"conf={confidence} pos={result.get('position','center')}")
            return {
                "match":        match_name,
                "relationship": rel,
                "confidence":   confidence,
                "position":     result.get("position", "center"),
            }
        except Exception as e:
            print(f"[person_id] parse error: {e}  raw={raw[:80]}")
            return None

    def _call_openrouter(self, model: str, img_b64: str,
                         find_target: Optional[str]) -> tuple:
        """Returns (success, http_status, content_or_error)."""
        if not self.api_key:
            return False, 0, "no openrouter api key"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://blindglasses.local",
            "X-Title":       "Blind Glasses Assistant",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": self.build_prompt(find_target)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ],
            "max_tokens":  1200,
            "temperature": 0.0,
        }
        try:
            r = requests.post(self.url, headers=headers,
                              json=payload, timeout=18)
            if r.status_code == 200:
                return True, 200, r.json()["choices"][0]["message"]["content"]
            print(f"[Perception] {model} → {r.status_code}: {r.text[:150]}")
            return False, r.status_code, r.text
        except Exception as e:
            print(f"[Perception] {model} request error: {e}")
            return False, 0, str(e)

    def _call_gemini(self, model_id: str, img_b64: str,
                     find_target: Optional[str]) -> tuple:
        """
        Call Google AI Studio (Gemini API) directly.

        Free tier (verified April 2026):
          - gemini-2.5-flash-lite : 15 RPM, 1000 RPD
          - gemini-2.5-flash      : 10 RPM,  250 RPD
          - gemini-2.5-pro        :  5 RPM,  100 RPD
        Resets at midnight Pacific time.  Vision included on free tier.
        """
        if not self.google_api_key:
            return False, 0, "no google api key"
        url = (f"{self.gemini_url}/{model_id}:generateContent"
               f"?key={self.google_api_key}")
        prompt_text = self._SYSTEM_PROMPT + "\n\n" + self.build_prompt(find_target)
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64,
                    }},
                ]
            }],
            "generationConfig": {
                "temperature":     0.0,
                "maxOutputTokens": 1200,
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=18,
                              headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                data = r.json()
                cands = data.get("candidates", [])
                if not cands:
                    return False, 200, "no candidates"
                parts = cands[0].get("content", {}).get("parts", [])
                text  = "".join(p.get("text", "") for p in parts)
                if not text:
                    return False, 200, "empty text"
                return True, 200, text
            print(f"[Perception] gemini/{model_id} → {r.status_code}: {r.text[:200]}")
            return False, r.status_code, r.text
        except Exception as e:
            print(f"[Perception] gemini/{model_id} request error: {e}")
            return False, 0, str(e)

    def is_quota_exhausted(self) -> bool:
        """
        True when every live VLM model is in a cooloff longer than
        1 hour — meaning the daily free-tier quota is gone for today.
        Frontend can use this to stop polling /vision entirely.
        """
        live = [m for m in self._models if m not in self._dead]
        if not live:
            return True
        now = time.time()
        return all((self._cooloff_until.get(m, 0) - now) > 3600 for m in live)

    # ── Output parser ─────────────────────────────────────────

    def parse_output(
        self,
        text: Optional[str],
        find_target: Optional[str] = None,
    ) -> dict:
        fallback = _empty_scene(find_target)
        if not text:
            return fallback

        try:
            clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
            m = (re.search(r"^\s*\{.*\}\s*$", clean, re.DOTALL) or
                 re.search(r"\{.*\}", clean, re.DOTALL))
            if not m:
                # No JSON at all — try to salvage info from prose.
                prose_scene = _parse_prose(clean, find_target)
                if prose_scene is not None:
                    print(f"[Perception] non-JSON salvaged → "
                          f"target_found={prose_scene.get('target_found')}")
                    return prose_scene
                raise ValueError("no JSON object")

            raw = m.group()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as je:
                # Truncated/malformed JSON — try common repairs:
                #   1. Trim from the last successfully-parsed comma
                #   2. Close any open brackets/braces
                repaired = _repair_truncated_json(raw)
                try:
                    data = json.loads(repaired)
                    print("[Perception] JSON repaired ✓")
                except Exception:
                    raise je

            data.setdefault("environment",        "unknown")
            data.setdefault("scene_type",         "other")
            data.setdefault("confidence",         "MEDIUM")
            data.setdefault("danger_level",       "LOW")
            data.setdefault("objects",            [])
            data.setdefault("immediate_threats",  [])
            data.setdefault("safe_path",          "center")
            data.setdefault("navigation_command", "GO_FORWARD")
            data.setdefault("guidance",           "")

            data["danger_level"] = str(data["danger_level"]).upper()
            if data["danger_level"] not in {"HIGH", "MEDIUM", "LOW"}:
                data["danger_level"] = "LOW"

            data["confidence"] = str(data["confidence"]).upper()
            if data["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
                data["confidence"] = "MEDIUM"

            # Sanitise object list, clamp bboxes
            sanitised = []
            for obj in data.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                sanitised.append({
                    "label":              str(obj.get("label", "unknown object")),
                    "position":           obj.get("position", "center"),
                    "distance":           obj.get("distance", "medium"),
                    "moving":             bool(obj.get("moving", False)),
                    "moving_toward_user": bool(obj.get("moving_toward_user", False)),
                    "threat":             bool(obj.get("threat", False)),
                    "bbox_cx": _clamp(obj.get("bbox_cx", 0.5), 0.0, 1.0),
                    "bbox_cy": _clamp(obj.get("bbox_cy", 0.5), 0.0, 1.0),
                    "bbox_w":  _clamp(obj.get("bbox_w",  0.25), 0.05, 1.0),
                    "bbox_h":  _clamp(obj.get("bbox_h",  0.20), 0.05, 1.0),
                })
            data["objects"] = sanitised

            # Target matching with synonym fallback (single-frame)
            if find_target:
                vlm_found = bool(data.get("target_found", False))
                if not vlm_found:
                    for obj in data["objects"]:
                        if _matches_target(obj["label"], find_target):
                            vlm_found = True
                            data["target_found"]    = True
                            data["target_position"] = obj["position"]
                            data["target_distance"] = obj["distance"]
                            data["target_bbox"]     = {
                                "cx": obj["bbox_cx"], "cy": obj["bbox_cy"],
                                "w":  obj["bbox_w"],  "h":  obj["bbox_h"],
                            }
                            obj["is_target"] = True
                            break
                if not vlm_found:
                    data.setdefault("target_found",    False)
                    data.setdefault("target_position", None)
                    data.setdefault("target_distance", None)
                    data.setdefault("target_bbox",     None)

            # Deterministic danger escalation (rules can only ESCALATE)
            forced_threats: list[str] = []
            for obj in data["objects"]:
                label    = obj["label"].lower()
                position = obj["position"]
                distance = obj["distance"]

                # Anything in NEVER_HAZARD (people, furniture, household
                # items, pets) is NOT escalated by the close+center rule.
                # It can still be HIGH if the VLM said it's actively
                # moving toward the user, or if it matches the explicit
                # hazard list (which never overlaps with these labels).
                is_safe_indoor = any(safe in label
                                     for safe in _NEVER_HAZARD_LABELS)

                is_explicit_hazard = any(h in label for h in _HAZARD_LABELS)
                is_close_center   = (distance == "close" and position == "center")
                is_moving_toward  = obj.get("moving_toward_user", False)

                # Decision: HIGH only when an actually-dangerous thing
                # is close, OR when something is moving at the user.
                if is_explicit_hazard or is_moving_toward:
                    obj["threat"]        = True
                    data["danger_level"] = "HIGH"
                    forced_threats.append(f"{obj['label']} — {distance} — {position}")
                elif is_close_center and not is_safe_indoor:
                    obj["threat"]        = True
                    data["danger_level"] = "HIGH"
                    forced_threats.append(f"{obj['label']} — {distance} — {position}")
                elif obj["threat"]:
                    forced_threats.append(f"{obj['label']} — {distance} — {position}")

            if data["danger_level"] == "LOW":
                for obj in data["objects"]:
                    label = obj["label"].lower()
                    if (obj["distance"] == "close"
                            and not any(s in label for s in _NEVER_HAZARD_LABELS)):
                        data["danger_level"] = "MEDIUM"
                        break

            data["immediate_threats"] = forced_threats or data["immediate_threats"]
            danger = data["danger_level"]
            data["is_dangerous"] = danger == "HIGH"

            if danger == "HIGH":
                cmds = {
                    "STOP":       "Stop immediately.",
                    "MOVE_LEFT":  "Move left now.",
                    "MOVE_RIGHT": "Move right now.",
                    "WAIT":       "Wait, do not move.",
                    "GO_FORWARD": "Caution ahead.",
                }
                guidance = data.get("guidance", "")
                cmd      = data.get("navigation_command", "GO_FORWARD")
                if guidance and len(guidance.split()) <= 14:
                    data["danger_message"] = guidance
                elif forced_threats:
                    data["danger_message"] = (
                        f"{cmds.get(cmd, 'Caution.')} "
                        f"{'. '.join(forced_threats[:2])}.")
                else:
                    data["danger_message"] = cmds.get(cmd, "Caution ahead.")
            else:
                data["danger_message"] = ""

            data["parse_error"] = False
            return data

        except Exception as e:
            fallback["parse_error_detail"] = str(e)
            fallback["raw_snippet"]        = (text or "")[:300]
            print(f"[Perception] parse error: {e}")
            return fallback

    # ── Temporal consensus ────────────────────────────────────

    def apply_consensus(self, scene: dict, find_target: Optional[str]) -> dict:
        """
        Update the rolling history for this target and override
        scene["target_found"] using the consensus rule.

          • FAST PATH: a single HIGH-confidence detection is enough
            to declare FOUND immediately (no 2/3 wait). This removes
            the lag the user was experiencing — when the camera is
            clearly pointed at the object, we react on frame 1.
          • Otherwise the rolling consensus rule applies:
              – target_found = True if at least
                VISION_MIN_CONFIRMATIONS of the last
                VISION_MAX_HISTORY frames detected the target.

        Sets scene["consensus_state"] for the orchestrator:
          "FOUND"     → speak success now
          "SEARCHING" → keep silent, the camera is still scanning
          "NOT_FOUND" → after enough negative scans, give a useful hint
        """
        if not find_target:
            return scene

        hist = self._history.setdefault(
            find_target, deque(maxlen=self._hist_size))
        single_frame_hit = bool(scene.get("target_found"))
        single_frame_conf = str(scene.get("confidence", "MEDIUM")).upper()
        hist.append(single_frame_hit)

        # ── Fast path: trust a single HIGH-confidence detection ──
        if single_frame_hit and single_frame_conf == "HIGH":
            scene["target_found"]    = True
            scene["consensus_state"] = "FOUND"
            return scene

        positives = sum(hist)
        if positives >= self._min_conf:
            scene["target_found"]    = True
            scene["consensus_state"] = "FOUND"
        elif len(hist) >= self._hist_size and positives == 0:
            scene["target_found"]    = False
            scene["consensus_state"] = "NOT_FOUND"
        else:
            scene["target_found"]    = False
            scene["consensus_state"] = "SEARCHING"
        return scene

    def reset_history(self, find_target: Optional[str] = None) -> None:
        if find_target is None:
            self._history.clear()
        else:
            self._history.pop(find_target, None)

    # ── Local cam loop helpers (unchanged) ────────────────────

    def process_frame(self, frame, find_target: Optional[str] = None):
        now = time.time()
        if now - self.last_request_time < self.request_delay:
            return self.last_result
        self.last_request_time = now
        frame_resized = cv2.resize(frame, (self.width, self.height))
        img_b64       = self.encode(frame_resized)
        raw           = self.ask(img_b64, find_target=find_target)
        scene         = self.parse_output(raw, find_target=find_target)
        scene         = self.apply_consensus(scene, find_target)
        self.last_result = {
            "timestamp":    now,
            "scene":        scene,
            "danger_level": scene.get("danger_level", "LOW"),
        }
        return self.last_result


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0