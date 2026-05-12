# WExist

> A mobile accessibility platform for blind and deaf users — built entirely on free infrastructure.

One Flutter app. Two complete subsystems. Zero paid subscriptions.

<!-- TODO: replace with a real demo GIF or screenshot -->
<p align="center">
  <img src="docs/demo.gif" alt="WExist demo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/LangGraph-FF6B35" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Blind subsystem](#blind-subsystem)
4. [Deaf subsystem](#deaf-subsystem)
5. [Mobile app](#mobile-app)
6. [Cost & free tier](#cost--free-tier)
7. [Setup](#setup)
8. [Project structure](#project-structure)

---

## What it does

WExist is a single Flutter app backed by a single Python server, serving two groups often underserved by mainstream tech:

**Blind / visually impaired users** — a voice-first AI companion.
- 🎙️ Talk to it in English, French, Arabic (MSA), or Tunisian Darija
- 👁️ Find objects, describe surroundings, identify saved people
- 🗺️ Turn-by-turn walking navigation with real-time obstacle warnings
- 🧠 Personal memory — "Where did I last see my keys?"

**Deaf / hard-of-hearing users** — a camera-first communication assistant.
- ✋ Real-time ASL fingerspelling (A–Z) and full word recognition
- 🤖 Type a sentence, watch a 3D avatar sign it
- 👄 Lip reading from the camera
- 🔔 Sound alarms (knock, baby cry, fire alarm) via vibration

**Key principles**

- **Voice-first** for blind users — every screen responds to speech, every reply is spoken.
- **Camera-first** for deaf users — sign language captured via camera, avatar renders text.
- **Fully free** — every model and API runs on a free tier or locally on CPU.
- **Multilingual** — auto-switches between EN / FR / AR / TN based on speech.
- **Private** — all personal data lives in local SQLite, scoped per device UUID. No accounts.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Flutter Mobile App                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │   Blind UX       │  │   Deaf UX        │  │  Shared Services  │  │
│  │  Voice + Camera  │  │  Sign + Avatar   │  │  Auth, Profile,   │  │
│  │  Navigation      │  │  Lip Reading     │  │  Alarm, TTS       │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬──────────┘  │
│           └────────────────────┴────────────────────────┘           │
│                                │ HTTP / WebSocket (LAN)             │
└────────────────────────────────┼────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                         │
│  REST + WebSocket + Static /signs/                                  │
│                                                                     │
│  LangGraph Orchestrator (blind subsystem only)                      │
│  → DialogueAgent · PerceptionAgent · NavigateAgent ·                │
│    StreetDangerAgent · EmotionAgent · ActionAgent                   │
│                                                                     │
│  Service Layer                                                      │
│  → Gesture · Sign · VSR · Memory · TextToSign · Alarm               │
│                                                                     │
│  Models & Free APIs                                                 │
│  → Whisper · YOLO · CTR-GCN · SigLIP · Chaplin VSR · Keras FER     │
│  → Gemini Flash-Lite · OpenRouter · Groq · Nominatim · OSRM         │
└─────────────────────────────────────────────────────────────────────┘
```

Backend runs as a single FastAPI process on the same LAN as the phone. No internet exposure required.

📖 **Full architecture details:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Blind subsystem

Voice-driven. User speaks → server interprets → response is spoken back. Camera frames are attached automatically when needed.

### Voice command flow

```
User holds button and speaks
        │
        ▼  POST /process  (WAV 16kHz + optional JPEG)
DialogueAgent — Whisper STT + LLM intent extraction
        │
        ▼
LangGraph state machine → agents run conditionally
        │
        ▼  JSON response → Flutter TTS → spoken to user
```

A LangGraph state machine routes the request through the right agents based on intent (find, describe, navigate, memory query, save, …).

### Agents

| Agent | Role |
|---|---|
| **DialogueAgent** | Entry point. Whisper STT + Llama-3.3-70b for intent extraction. Handles local keyword shortcuts in all 4 languages and YES/NO confirmation flows. |
| **PerceptionAgent** | Vision via a VLM cascade (Gemini Flash-Lite → Flash → Pro → OpenRouter VLMs). Uses temporal consensus across 3 frames to avoid hallucinations. Also identifies saved people. |
| **NavigateAgent** | Walking routes via Overpass (POI search) + Nominatim (named places) + OSRM (turn-by-turn). Detects off-route and recalculates. |
| **StreetDangerAgent** | Custom YOLOv8 detecting street obstacles (cars, potholes, barriers, steps) during navigation. Interrupts ongoing speech. |
| **EmotionAgent** | Keras FER classifying 7 emotions on detected faces. Active in describe mode only. |
| **ActionAgent** | Final node. Assembles multilingual speech and deduplicates repeated messages (danger alerts always pass through). |

### Personal memory

Stored in a local SQLite file scoped per device UUID — no accounts, no logins.

| Table | What it holds |
|---|---|
| `object_sightings` | Auto-logged finds: name, GPS, scene description, timestamp |
| `personal_objects` | User's belongings with descriptions and images |
| `personal_persons` | Saved people: name, relationship, face description |
| `named_locations` | Pinned places: home, work, pharmacy |
| `user_profile` | User's name and settings |

Every successful find is auto-logged. *"Where did I last see my keys?"* queries the most recent sighting, reverse-geocodes the GPS, and speaks a relative timestamp.

### Navigation

Saying *"Take me to the pharmacy"* triggers POI search via Overpass at increasing radii, then OSRM routing. During the trip, GPS polls every ~10 s for turn instructions, and the camera streams to the danger agent in parallel.

### Endpoints

- **REST:** `/process`, `/vision`, `/navigate`, `/nav_danger`, `/describe_for_save[_person]`, `/save_personal_object`, `/save_person`, `/user/name`, `/user/profile`, `/user/{persons|objects|locations}/{id}` (DELETE), `/profile/command`, `/sign/text_to_sign`
- **WebSocket:** `/ws/gesture`, `/ws/sign`, `/ws/vsr`, `/ws/alarm`

---

## Deaf subsystem

No voice, no LangGraph. Four independent modules, each with its own endpoint. All processing local.

| Module | Endpoint | Model | What it does |
|---|---|---|---|
| **ASL fingerspelling** | `/ws/gesture` | SigLIP ViT (99.96% on ASL alphabet) | Recognizes A–Z hand signs frame by frame. Majority vote across 20 frames, Groq builds the sentence from accumulated letters. |
| **Full ASL words** | `/ws/sign` | CTR-GCN (skeleton-based) | Recognizes full word signs from MediaPipe Holistic landmarks (288-dim feature vector over 60 frames). |
| **Text to sign** | `POST /sign/text_to_sign` | NLTK + video library | Tokenizes input, lemmatizes, detects tense (ASL grammar), plays a 3D avatar video per word, spells letter-by-letter as fallback. |
| **Lip reading** | `/ws/vsr` | Chaplin VSR (LRS3 trained, ~19% WER) | Reads lips from MediaPipe face landmarks → ResNet encoder → Transformer decoder. Optional Groq refinement. |
| **Alarm system** | `/ws/alarm` | YAMNet-based classifier | Monitors ambient audio for knocks, baby cries, fire alarms. Scheduled alarms fire as vibration + visual alerts. |

The sign video library has **151 MP4s**: 125 word videos + 26 letter videos. All four WebSocket endpoints share the same JSON protocol (`ready`, `status`, `letter`/`word`/`sentence`/`transcription`, `error`).

---

## Mobile app

```
App Launch
    │
    ▼
BiometricAuthScreen  (fingerprint / PIN)
    │
    ▼
HomePage  (animated orb + voice button)
    │
    ├── Blind Features
    │   ├── Voice Command       → /process
    │   ├── Camera Screen       → /vision every 4s
    │   ├── Navigation Screen   → /navigate + /nav_danger + OSM map
    │   ├── Save Object/Person  → describe + confirm + save
    │   ├── Lip Reading         → /ws/vsr
    │   ├── Handwriting OCR
    │   └── Profile             → /user/profile + /profile/command
    │
    └── Deaf Features
        ├── Sign Language       → /ws/gesture
        ├── Text to Sign        → POST /sign/text_to_sign
        ├── Alarm               → /ws/alarm
        └── Profile             (shared)
```

**Shared services:** `ApiService` (HTTP/WS, set `baseUrl` here), `CameraService`, `Recorder` (16 kHz WAV), `LocationService` (Geolocator), `TtsService` (4 languages), `AlarmService`.

**User identity:** UUID generated on first launch, stored in `SharedPreferences`, sent as `X-User-ID` on every request — all memory is scoped to it.

**Key packages:** `camera` · `record` · `http` · `web_socket_channel` · `flutter_tts` · `geolocator` · `flutter_map` · `video_player` · `local_auth` · `shared_preferences` · `vibration`

---

## Cost & Free Tier

**$0/month.** Every component is free.

| Component | Provider |
|---|---|
| Speech-to-text | faster-whisper (local) |
| Intent LLM | OpenRouter — Llama-3.3-70B:free |
| Vision (primary) | Gemini 2.5 Flash-Lite |
| Vision (fallback) | Gemini 2.5 Flash → Pro → OpenRouter VLMs |
| Sentence building | Groq — Llama-3.3-70B |
| Geocoding / POI / routing | Nominatim · Overpass · OSRM |
| Map tiles | OpenStreetMap |
| Vision / sign / lip models | YOLO · SigLIP · CTR-GCN · Chaplin · Keras FER (local CPU) |
| Personal memory | SQLite local database |

At a **4-second scan interval**, Gemini Flash-Lite’s **1,000 requests/day** allowance provides roughly **67 minutes of continuous camera-on usage** before automatic fallback models are used.

---

## Setup

**Requirements:** Python 3.9+, Flutter 3.x, phone and PC on the same WiFi.

### 1. Free API keys (no credit card)

| Key | Source |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

### 2. Backend

```bash
cd voice-assistant/backend
pip install -r requirements.txt
export GOOGLE_API_KEY="..." OPENROUTER_API_KEY="..." GROQ_API_KEY="..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Auto-downloaded on first run: Whisper small (~460 MB), SigLIP ViT (~400 MB), MediaPipe task files (~20 MB each).

**Manual model files** (server runs fine without them — features silently disabled):

| File | Path |
|---|---|
| YOLOv8 weights | `backend/models/yolo/best.pt` |
| Keras FER | `backend/models/emotion/best.h5` |
| CTR-GCN | `backend/models/sign/ctr-gcn/` |

### 3. Flutter

```bash
cd voice-assistant/mobile
flutter pub get
# Edit lib/api_service.dart → baseUrl = "http://<your-PC-IP>:8000"
flutter run
```

⚠️ Full restart (not hot reload) required after adding native packages.

---

## Project structure

```
wexist/
├── backend/
│   ├── main.py                  # FastAPI entry + model boot
│   ├── config.py                # paths, keys, thresholds
│   ├── agents/
│   │   ├── DialogueAgent.py     # Whisper STT + LLM intent
│   │   ├── PerceptionAgent.py   # VLM cascade + consensus
│   │   ├── NavigateAgent.py     # OSM/OSRM routing
│   │   ├── StreetDangerAgent.py # YOLOv8 obstacles
│   │   ├── EmotionAgent.py      # Keras FER
│   │   ├── ActionAgent.py       # Speech dedup
│   │   └── orchestrator_graph.py # LangGraph wiring
│   ├── gesture_service.py       # ASL fingerspelling (SigLIP)
│   ├── sign_service.py          # Full ASL words (CTR-GCN)
│   ├── text_to_sign_service.py  # Text → sign videos
│   ├── vsr_service.py           # Lip reading (Chaplin)
│   ├── sound_alarm_service.py   # Alarms + sound monitoring
│   ├── memory_service.py        # SQLite CRUD
│   ├── phrasebook.py            # 100+ phrases × 4 languages
│   └── static/signs/            # 151 sign MP4s (125 words + 26 letters)
│
└── mobile/lib/
    ├── main.dart                # → BiometricAuthScreen
    ├── ui_home.dart             # Animated orb + voice button
    ├── camera_screen.dart       # Continuous vision loop
    ├── navigation_screen.dart   # OSM map + turn-by-turn
    ├── gesture_sign_page.dart   # ASL fingerspelling
    ├── sign_language_page.dart  # Full ASL words
    ├── text_to_sign_page.dart   # 3D avatar signer
    ├── lip_reading_page.dart    # Chaplin VSR
    ├── alarm_page.dart          # Alarms
    ├── handwriting_screen.dart  # OCR
    ├── save_object_screen.dart  # Save flows
    ├── save_person_screen.dart
    ├── profile_page.dart        # Manage memory
    └── api_service.dart         # ← set baseUrl here
```

---

## Roadmap

- [ ] iOS support (Android-first currently)
  

---


---

*WExist — because existing in the world should not depend on which senses you were born with.*
