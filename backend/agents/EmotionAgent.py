"""
EmotionAgent.py — local emotion recognition (free, offline)
─────────────────────────────────────────────────────────────────
Runs a pre-trained ResNet50 (.h5 Keras model) on detected faces
and returns one of seven emotions:

   angry, disgust, fear, happy, sad, surprise, neutral

DESIGN:

  • Free: model runs locally on CPU, no API calls.
  • Standalone: face detection uses OpenCV's Haar cascade which
    ships with cv2 — no extra dependency.
  • Privacy-respectful: emotions are only inferred when the
    orchestrator's DESCRIBE node calls us (i.e. the user explicitly
    asked "describe what's around me").  We never run during
    navigation or background scanning.
  • Honest: we phrase outputs as "looks happy" / "appears sad",
    never as authoritative claims about a stranger's feelings.
  • Graceful: if the .h5 model file is missing, the agent stays
    in `ready=False` mode and the orchestrator skips it silently.

USAGE:
    agent = EmotionAgent(model_path="models/emotion/best.h5")
    if agent.ready:
        results = agent.detect(frame_bgr)
        # → [{label, confidence, position, bbox}, ...]
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


# Standard FER-2013 / AffectNet emotion order.  If your model was
# trained with a different order, override via the constructor arg.
DEFAULT_EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
]


class EmotionAgent:
    """
    Loads a Keras .h5 emotion classifier once, then on each call
    detects faces in a frame and returns the predicted emotion
    for each face, along with position (left/center/right) and
    bbox so the UI can draw it.
    """

    def __init__(
        self,
        model_path: str = "models/emotion/best.h5",
        labels: Optional[list[str]] = None,
        conf_threshold: float = 0.45,
        # Face detector params.  Defaults tuned to catch faces in
        # typical living-room conditions: medium light, slight head
        # turn, smaller face (further from camera).  False positives
        # are filtered by the emotion confidence threshold.
        haar_scale_factor: float = 1.1,
        haar_min_neighbors: int = 4,
        min_face_size_px: int = 40,
    ):
        self.model_path     = model_path
        self.labels         = labels or DEFAULT_EMOTION_LABELS
        self.conf_threshold = conf_threshold
        self.haar_scale     = haar_scale_factor
        self.haar_neighbors = haar_min_neighbors
        self.min_face       = min_face_size_px

        self.ready: bool = False
        self.model        = None
        self.input_h: int = 224     # auto-detected after load
        self.input_w: int = 224
        self.input_c: int = 3       # 1 = grayscale, 3 = RGB
        self.face_cascade    = None
        self.profile_cascade = None

        self._try_load()

    # ── Loading ───────────────────────────────────────────────

    def _try_load(self):
        """Attempt to load Keras model + Haar cascade.  Failures are
        non-fatal — agent stays in ready=False mode."""
        if not os.path.exists(self.model_path):
            print(f"[EmotionAgent] model not found at {self.model_path} — "
                  f"emotion detection DISABLED")
            return

        # Lazy-import TF/Keras to avoid loading it for users who
        # don't enable the emotion feature.
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter
            import tensorflow as _tf
            load_model = _tf.keras.models.load_model
        except Exception as e:
            print(f"[EmotionAgent] tensorflow not installed ({e}) — "
                  f"DISABLED.  Install with: pip install tensorflow")
            return

        try:
            print(f"[EmotionAgent] loading {self.model_path}…")
            self.model = load_model(self.model_path, compile=False)
        except Exception as e:
            print(f"[EmotionAgent] could not load model: {e}")
            return

        # Auto-detect input shape from the model itself
        try:
            shape = self.model.input_shape  # e.g. (None, 224, 224, 3)
            if isinstance(shape, list):
                shape = shape[0]
            # shape is (batch, H, W, C)
            self.input_h = int(shape[1])
            self.input_w = int(shape[2])
            self.input_c = int(shape[3])
            print(f"[EmotionAgent] input shape: "
                  f"{self.input_h}x{self.input_w}x{self.input_c}")
        except Exception as e:
            print(f"[EmotionAgent] could not read input shape ({e}), "
                  f"defaulting to 224x224x3")

        # Load OpenCV's bundled face cascades (no download needed).
        # The frontal cascade is the primary detector; the profile
        # cascade is run as a fallback so we catch faces turned away
        # from the camera too.
        try:
            cascade_dir = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
            self.profile_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_dir, "haarcascade_profileface.xml"))
            if self.face_cascade.empty():
                print("[EmotionAgent] Haar frontal cascade failed to load")
                return
            if self.profile_cascade.empty():
                print("[EmotionAgent] Haar profile cascade missing — frontal only")
                self.profile_cascade = None
        except Exception as e:
            print(f"[EmotionAgent] cascade error: {e}")
            return

        self.ready = True
        print(f"[EmotionAgent] ready ✓ (labels: {self.labels})")

    # ── Inference ─────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces and predict emotion for each.

        Args:
            frame_bgr: BGR image as numpy array (HxWx3, uint8) — i.e.
                       the raw cv2 / cv2.imdecode output.

        Returns:
            List of dicts, ordered by face area (largest first):
              {
                "label":      "happy",
                "confidence": 0.87,
                "position":   "center" | "left" | "right",
                "bbox":       {"cx", "cy", "w", "h"}   # normalised 0-1
              }
            Returns [] if not ready, no faces, or low confidence.
        """
        if not self.ready or frame_bgr is None or frame_bgr.size == 0:
            return []

        try:
            h, w = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Equalise histogram to handle uneven lighting (Haar is
            # very sensitive to contrast)
            gray = cv2.equalizeHist(gray)

            # Detect faces (Haar — fast, no GPU needed).  Try the
            # frontal cascade first; if nothing is found, fall back to
            # the profile cascade for side-facing faces.
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.haar_scale,
                minNeighbors=self.haar_neighbors,
                minSize=(self.min_face, self.min_face),
            )
            if len(faces) == 0 and self.profile_cascade is not None:
                faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.haar_scale,
                    minNeighbors=self.haar_neighbors,
                    minSize=(self.min_face, self.min_face),
                )
                # Also try mirrored frame for the OTHER profile
                # (Haar profile cascade only detects right-facing
                # profiles).
                if len(faces) == 0:
                    flipped = cv2.flip(gray, 1)
                    faces_flipped = self.profile_cascade.detectMultiScale(
                        flipped,
                        scaleFactor=self.haar_scale,
                        minNeighbors=self.haar_neighbors,
                        minSize=(self.min_face, self.min_face),
                    )
                    # Mirror coordinates back
                    faces = [(w - x - fw, y, fw, fh)
                             for (x, y, fw, fh) in faces_flipped]
            if len(faces) == 0:
                return []

            # Sort by area, largest first (closest face first)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            results = []
            # Cap at 3 faces per frame to keep latency low
            for (x, y, fw, fh) in faces[:3]:
                # Crop face from the colour frame for the model
                # (most ResNet50 emotion models expect colour input)
                face_crop = frame_bgr[y:y+fh, x:x+fw]
                if face_crop.size == 0:
                    continue

                pred = self._predict_one(face_crop)
                if pred is None:
                    continue
                label, conf = pred
                if conf < self.conf_threshold:
                    continue

                # Position from face centroid
                cx_pix = x + fw / 2.0
                if cx_pix < w / 3:
                    position = "left"
                elif cx_pix > 2 * w / 3:
                    position = "right"
                else:
                    position = "center"

                results.append({
                    "label":      label,
                    "confidence": float(conf),
                    "position":   position,
                    "bbox": {
                        "cx": cx_pix / w,
                        "cy": (y + fh / 2.0) / h,
                        "w":  fw / w,
                        "h":  fh / h,
                    },
                })
            return results
        except Exception as e:
            print(f"[EmotionAgent] detect error: {e}")
            return []

    def _predict_one(self, face_bgr: np.ndarray) -> Optional[tuple[str, float]]:
        """Preprocess one face and run the Keras model."""
        try:
            # Resize to model's expected H×W
            face = cv2.resize(face_bgr, (self.input_w, self.input_h),
                              interpolation=cv2.INTER_AREA)

            # Channel handling: model wants 1 or 3 channels
            if self.input_c == 1:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = face[..., np.newaxis]   # (H,W) → (H,W,1)
            else:
                # Most Keras emotion models expect RGB, not BGR
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Normalise to [0, 1].  If your model needs ImageNet
            # mean/std subtraction, change this block.
            arr = face.astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)   # add batch dim

            probs = self.model.predict(arr, verbose=0)[0]
            idx   = int(np.argmax(probs))
            if idx < 0 or idx >= len(self.labels):
                return None
            return self.labels[idx], float(probs[idx])
        except Exception as e:
            print(f"[EmotionAgent] predict error: {e}")
            return None