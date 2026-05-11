"""
backend/vsr_context.py
═══════════════════════
Context analysis and translation for VSR (lip reading) output.
Uses Groq Llama to understand intent, generate action cards, and translate.
"""

import json
from typing import Optional
from groq import Groq

_client: Optional[Groq] = None


def init(api_key: str):
    global _client
    if api_key and not _client:
        try:
            _client = Groq(api_key=api_key)
            print("[vsr_context] Groq context client ready")
        except Exception as e:
            print(f"[vsr_context] Groq init failed: {e}")


def analyze(text: str) -> dict:
    """Analyze VSR output text; return intent + action cards."""
    if not _client or not text.strip() or text.startswith("["):
        return {"intent": text, "actions": _default_actions(text)}

    prompt = f"""You analyze lip-reading output and generate action cards.

Lip-read text: "{text}"

Return a JSON object:
{{
  "intent": "<one-sentence description of what the person meant>",
  "actions": [ ... ]
}}

ACTION CARD RULES — read carefully:

1. COPY — ALWAYS include:
   {{"icon":"📋","label":"Copy","type":"copy","payload":"{text}"}}

2. TRANSLATE intent — e.g. "translate hello in arabic", "how do you say good morning in french", "what is thank you in arabic":
   Extract ONLY the word/phrase to translate (NOT the full sentence).
   {{"icon":"🌍","label":"Translate","type":"translate","payload":{{"text":"<ONLY the word/phrase>","language":"<target language lowercase>"}}}}
   Example: "translate hello in arabic" → payload: {{"text":"hello","language":"arabic"}}
   Example: "how do you say good morning in french" → payload: {{"text":"good morning","language":"french"}}

3. HUNGRY / FOOD / EATING / DELIVERY:
   {{"icon":"🛵","label":"Open Glovo","type":"glovo","payload":"glovo"}}

4. NAVIGATION / GOING SOMEWHERE / WANT TO GO TO <place>:
   Extract the destination from the sentence.
   {{"icon":"🗺️","label":"Navigate","type":"maps","payload":"<destination>"}}
   Example: "I want to go to the pharmacy" → payload: "pharmacy"
   Example: "take me to the hospital" → payload: "hospital"

5. FACTUAL QUESTION / INFORMATION:
   {{"icon":"🔍","label":"Search","type":"web","payload":"<concise search query>"}}

Generate only the action cards that genuinely fit. Do NOT invent unrelated cards.
Return ONLY valid JSON, no markdown."""

    try:
        resp = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        # Guarantee copy action is always present
        types = [a.get("type") for a in data.get("actions", [])]
        if "copy" not in types:
            data.setdefault("actions", []).append(
                {"icon": "📋", "label": "Copy", "type": "copy", "payload": text}
            )
        return data
    except Exception as e:
        print(f"[vsr_context] analyze failed: {e}")
        return {"intent": text, "actions": _default_actions(text)}


_LANG_NAMES = {
    "french": "French",
    "arabic": "Modern Standard Arabic",
    "darija": "Tunisian Darija (Tunisian Arabic dialect, written in Arabic script)",
}


def translate(text: str, target_language: str) -> str:
    """Translate text to target language using Groq."""
    if not _client or not text.strip():
        return ""

    lang_name = _LANG_NAMES.get(target_language.lower(), target_language)

    try:
        resp = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following English text to {lang_name}.\n"
                    f"Return ONLY the translation, no explanation.\n\n"
                    f"Text: {text}"
                ),
            }],
            temperature=0.1,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[vsr_context] translate failed: {e}")
        return ""


def _default_actions(text: str) -> list:
    return [{"icon": "📋", "label": "Copy", "type": "copy", "payload": text}]
