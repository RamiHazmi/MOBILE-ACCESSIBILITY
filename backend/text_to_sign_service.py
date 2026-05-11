"""
text_to_sign_service.py
─────────────────────────────────────────────────────────────────
Converts plain English text into a sequence of sign-language tokens.

Pipeline
────────
  1. Tokenise + lowercase + strip punctuation
  2. Detect tense → prepend "BEFORE" (past) or "WILL" (future)
  3. NLTK lemmatisation + optional stopword removal
  4. Each token is resolved to:
       a. A whole-word video file   (backend/static/signs/<WORD>.mp4)
       b. Letter-by-letter fallback (backend/static/signs/letters/<X>.mp4)

Install NLTK resources once:
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet');
               nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SIGNS_DIR   = os.path.join(BASE_DIR, "static", "signs")
LETTERS_DIR = os.path.join(SIGNS_DIR, "letters")

# ── NLTK bootstrap ───────────────────────────────────────────────────
_nltk_ok    = False
_lemmatizer = None
_stopwords_en: set = set()

# Words we never strip even if NLTK marks them as stopwords
_KEEP = {
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "no", "not", "yes", "please", "help", "need", "want",
    "is", "am", "are", "was", "were", "do", "did", "does",
    "will", "can", "have", "has", "had", "go", "come", "stop",
}

def _init_nltk() -> None:
    global _nltk_ok, _lemmatizer, _stopwords_en
    try:
        import nltk
        for pkg in ("punkt", "punkt_tab", "wordnet", "stopwords",
                    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                    "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords as sw
        _lemmatizer   = WordNetLemmatizer()
        _stopwords_en = set(sw.words("english")) - _KEEP
        _nltk_ok = True
        print("[text_to_sign] NLTK ready")
    except Exception as exc:
        print(f"[text_to_sign] NLTK unavailable ({exc}) — using basic tokeniser")

_init_nltk()


# ── Tense detection ──────────────────────────────────────────────────
_PAST_WORDS   = {"yesterday", "ago", "last", "before", "previously",
                 "was", "were", "had", "did", "went", "came"}
_FUTURE_WORDS = {"tomorrow", "will", "shall", "next", "soon", "later",
                 "going", "gonna", "plan"}

def _tense_marker(words: list[str]) -> Optional[str]:
    low = {w.lower() for w in words}
    if low & _FUTURE_WORDS:
        return "WILL"
    if low & _PAST_WORDS:
        return "BEFORE"
    return None


# ── NLTK POS → WordNet POS ───────────────────────────────────────────
def _wn_pos(tag: str) -> str:
    from nltk.corpus import wordnet
    return (wordnet.ADJ  if tag.startswith("J") else
            wordnet.ADV  if tag.startswith("R") else
            wordnet.VERB if tag.startswith("V") else
            wordnet.NOUN)


# ── Tokenisation ─────────────────────────────────────────────────────
def text_to_tokens(text: str, remove_stopwords: bool = False) -> list[str]:
    """
    Convert free-form English into an ordered list of uppercase sign tokens.
    """
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if _nltk_ok:
        import nltk
        words  = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(words)
        marker = _tense_marker(words)

        tokens: list[str] = []
        if marker:
            tokens.append(marker)

        for word, tag in tagged:
            if not re.match(r"^[a-zA-Z'-]+$", word):
                continue
            word_l = word.lower()
            if remove_stopwords and word_l in _stopwords_en:
                continue
            lemma = _lemmatizer.lemmatize(word_l, pos=_wn_pos(tag))  # type: ignore[arg-type]
            tokens.append(lemma.upper())
    else:
        words  = text.split()
        marker = _tense_marker(words)
        tokens = [marker] if marker else []
        tokens += [
            re.sub(r"[^A-Z]", "", w.upper())
            for w in words
            if re.match(r"^[a-zA-Z]+$", w)
        ]

    return [t for t in tokens if t]


# ── Video resolution ─────────────────────────────────────────────────
def _word_video_exists(token: str) -> bool:
    return os.path.isfile(os.path.join(SIGNS_DIR, f"{token.upper()}.mp4"))

def _letter_video_exists(ch: str) -> bool:
    return os.path.isfile(os.path.join(LETTERS_DIR, f"{ch.upper()}.mp4"))


def build_sign_sequence(text: str, base_url: str,
                        remove_stopwords: bool = False) -> list[dict]:
    """
    Main entry point.

    Returns a list of segment dicts:
      {
        "token":     "HELLO",
        "type":      "word",
        "video_url": "http://…/static/signs/HELLO.mp4" | null,
        "letters": [
          {"letter": "H", "video_url": "…" | null},
          ...
        ]
      }
    Callers should flatten `letters` for letter-by-letter playback
    when `video_url` is null.
    """
    os.makedirs(SIGNS_DIR,   exist_ok=True)
    os.makedirs(LETTERS_DIR, exist_ok=True)

    tokens   = text_to_tokens(text, remove_stopwords=remove_stopwords)
    base_url = base_url.rstrip("/")
    segments: list[dict] = []

    for token in tokens:
        if _word_video_exists(token):
            segments.append({
                "token":     token,
                "type":      "word",
                "video_url": f"{base_url}/static/signs/{token.upper()}.mp4",
                "letters":   [],
            })
        else:
            letters = []
            for ch in token:
                if ch.isalpha():
                    has_v = _letter_video_exists(ch)
                    letters.append({
                        "letter":    ch.upper(),
                        "video_url": (
                            f"{base_url}/static/signs/letters/{ch.upper()}.mp4"
                            if has_v else None
                        ),
                    })
            segments.append({
                "token":     token,
                "type":      "word",
                "video_url": None,
                "letters":   letters,
            })

    return segments
