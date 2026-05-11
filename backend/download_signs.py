"""
download_signs.py
─────────────────
One-time script: downloads all 151 ISL avatar MP4s from
  https://github.com/jigargajjar55/Audio-Speech-To-Sign-Language-Converter

Saves them into:
  backend/static/signs/letters/  ← A.mp4 … Z.mp4
  backend/static/signs/          ← HELLO.mp4, THANK.mp4, …  (uppercase)

Run once:
  cd voice-assistant/backend
  python download_signs.py
"""

import json
import os
import sys
import time
import urllib.request

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SIGNS_DIR   = os.path.join(BASE_DIR, "static", "signs")
LETTERS_DIR = os.path.join(SIGNS_DIR, "letters")

API_URL  = ("https://api.github.com/repos/"
            "jigargajjar55/Audio-Speech-To-Sign-Language-Converter"
            "/contents/assets")
RAW_BASE = ("https://raw.githubusercontent.com/"
            "jigargajjar55/Audio-Speech-To-Sign-Language-Converter"
            "/master/assets/")

os.makedirs(SIGNS_DIR,   exist_ok=True)
os.makedirs(LETTERS_DIR, exist_ok=True)

def fetch_json(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

def download(url: str, dest: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

print("Fetching file list from GitHub API…")
try:
    files = fetch_json(API_URL)
except Exception as e:
    print(f"GitHub API error: {e}")
    print("Falling back to known word list…")
    files = []

mp4_files = [f for f in files if isinstance(f, dict) and f.get("name", "").endswith(".mp4")]

if not mp4_files:
    # Hard-coded fallback if API is rate-limited
    KNOWN = (
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
        "0 1 2 3 4 5 6 7 8 9 "
        "After Afternoon Again Age All Aunty Before Beautiful Boy Can Come Day "
        "Do Does Done Dont Eat Engineer Evening Finish Friend Girl Give Go Good "
        "Have Hello Help Home I Invent Like Man Me Morning Mother Name "
        "Next No Not Now Please Right Same School Sorry Stop "
        "Talk Thank Then There This Time Today Tomorrow Trust Uncle Walk Want "
        "What When Where Who Will With Woman Yes You"
    ).split()
    mp4_files = [{"name": f"{w}.mp4"} for w in KNOWN]
    print(f"Using {len(mp4_files)} known files.")
else:
    print(f"Found {len(mp4_files)} MP4 files.")

ok = 0
skip = 0
fail = 0

for entry in mp4_files:
    name  = entry["name"]           # e.g. "Hello.mp4" or "A.mp4"
    stem  = name[:-4]               # strip .mp4

    # Decide destination
    if len(stem) == 1 and stem.isalpha():
        # Single letter → letters/
        dest = os.path.join(LETTERS_DIR, f"{stem.upper()}.mp4")
        label = f"letter/{stem.upper()}"
    else:
        # Word / phrase / digit → signs/ (uppercase)
        dest = os.path.join(SIGNS_DIR, f"{stem.upper()}.mp4")
        label = f"word/{stem.upper()}"

    if os.path.exists(dest):
        print(f"  skip  {label} (already downloaded)")
        skip += 1
        continue

    raw_url = RAW_BASE + urllib.request.quote(name)
    print(f"  dl {label}  ({raw_url.split('/')[-1]})")
    if download(raw_url, dest):
        ok += 1
    else:
        fail += 1

    time.sleep(0.05)   # be polite to GitHub

print(f"\nDone.  ok={ok}  skipped={skip}  failed={fail}")
print(f"Signs dir: {SIGNS_DIR}")
print(f"Letters dir: {LETTERS_DIR}")
