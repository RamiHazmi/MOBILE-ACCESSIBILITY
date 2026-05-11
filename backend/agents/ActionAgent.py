"""
ActionAgent.py — v4 (server-side logger / verifier)
─────────────────────────────────────────────────────
On a Nokia 3.2 + Flutter front-end, all real TTS is performed
on-device using the free Android TTS engine (flutter_tts).  The
backend ActionAgent is therefore a lightweight wrapper that:

  • logs what is going to be spoken
  • records the last spoken message for debugging / replay
  • optionally drives a local Piper voice for desktop testing

Local Piper is OPTIONAL — if the .onnx voices are absent, the
agent gracefully degrades to logging only.  This keeps the whole
backend free of paid services.
"""

import os
import io
import time
import wave
import threading


class ActionAgent:

    def __init__(self):
        self.is_speaking = False
        self._lock       = threading.Lock()
        self._stop_flag  = False
        self.last_message: str = ""
        # Dedup state — same text within the cooldown window is dropped
        self._last_text:    str   = ""
        self._last_text_ts: float = 0.0
        self._dedup_cooldown_s: float = 5.0

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ar_voice = os.path.join(BASE_DIR, "models/tts/ar_JO-kareem-medium.onnx")
        self.voices = {
            "en": os.path.join(BASE_DIR, "models/tts/en_US-lessac-medium.onnx"),
            "fr": os.path.join(BASE_DIR, "models/tts/fr_FR-siwis-medium.onnx"),
            "ar": ar_voice,
            "tn": ar_voice,   # Tunisian Derja uses the same Arabic voice
        }
        self.loaded_voices: dict = {}

        # Optional desktop playback — try to load Piper + pygame
        self._piper_ok = False
        try:
            from piper import PiperVoice  # noqa: F401
            import pygame                  # noqa: F401
            self._piper_ok = True
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16,
                                  channels=1, buffer=512)
            except Exception:
                self._piper_ok = False
            print("[ActionAgent] Piper + pygame available")
        except ImportError:
            print("[ActionAgent] Piper not available — log-only mode "
                  "(Flutter TTS handles speech on-device, free)")

    # ── Public API ────────────────────────────────────────────

    def speak(self, text: str, lang: str = "en", interrupt: bool = False):
        if not text:
            return
        # Defensive: upstream may accidentally pass a tuple/list or
        # non-string (e.g. a (str,) tuple from a comma typo in a
        # template).  Coerce to str so the dedup / print never crashes.
        if not isinstance(text, str):
            try:
                if isinstance(text, (list, tuple)) and len(text) > 0:
                    text = str(text[0])
                else:
                    text = str(text)
            except Exception:
                return
        # Dedup — drop identical lines repeated too fast (NEVER drop
        # interrupts, those are danger warnings)
        now = time.time()
        if (not interrupt
                and text.strip() == self._last_text.strip()
                and (now - self._last_text_ts) < self._dedup_cooldown_s):
            print(f"[TTS] (dedup-skip) ({lang}) {text}")
            return
        self._last_text    = text
        self._last_text_ts = now

        self.last_message = text
        tag = "⚡ INTERRUPT — " if interrupt else ""
        print(f"[TTS] {tag}({lang}) {text}")

        if not self._piper_ok:
            return

        def _run():
            with self._lock:
                if interrupt:
                    self._stop_flag = True
                    try:
                        import pygame
                        pygame.mixer.stop()
                    except Exception:
                        pass
                    time.sleep(0.05)
                    self._stop_flag = False
                self.is_speaking = True
                try:
                    wav_bytes = self._synth(text, lang)
                    self._play(wav_bytes)
                except FileNotFoundError:
                    # Piper voice file not installed — expected on the
                    # free setup where Flutter TTS handles speech on
                    # the phone.  Stay quiet to avoid log spam.
                    pass
                except Exception as e:
                    print(f"[ActionAgent] TTS error: {e}")
                finally:
                    self.is_speaking = False

        threading.Thread(target=_run, daemon=True).start()

    def speak_danger(self, message: str):
        if not message:
            return
        print(f"[ActionAgent] 🚨 DANGER {message}")
        self.speak(message, lang="en", interrupt=True)

    def speak_normal(self, text: str, lang: str = "en"):
        self.speak(text, lang=lang, interrupt=False)

    def stop(self):
        self._stop_flag = True
        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass
        self.is_speaking = False
        time.sleep(0.05)
        self._stop_flag = False

    # ── Piper helpers ─────────────────────────────────────────

    def _synth(self, text: str, lang: str) -> bytes:
        from piper import PiperVoice
        if lang not in self.voices:
            lang = "en"
        if lang not in self.loaded_voices:
            path = self.voices[lang]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Voice model not found: {path}")
            self.loaded_voices[lang] = PiperVoice.load(path)
        voice = self.loaded_voices[lang]
        buf   = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return buf.getvalue()

    def _play(self, wav_bytes: bytes):
        import pygame
        sound   = pygame.mixer.Sound(io.BytesIO(wav_bytes))
        channel = sound.play()
        while channel.get_busy():
            if self._stop_flag:
                channel.stop()
                break
            time.sleep(0.05)