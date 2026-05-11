"""
backend/sign_service.py
═══════════════════════════════════════════════════════════════
ASL Sign Language interpreter using a local CTR-GCN skeleton model.

Pipeline (now matches webcam_predict.py exactly):
  1. MediaPipe Tasks HolisticLandmarker (>= 0.10.30)
       → 33 pose (×4: x,y,z,vis) + 21 left-hand (×3) + 21 right-hand (×3)
         + 10 sparse face (×3)  =  132 + 63 + 63 + 30 = 288 features / frame
  2. Accumulate frames → shape (T, 288)
  3. Pad/truncate to (T_MAX, 288) then normalise with train_mean/train_std
  4. Model forward with input (1, T, 288)   ← same as webcam_predict.py
  5. Groq LLaMA → build natural sentence (optional, falls back gracefully)

Model directory layout (same as webcam_predict.py expects):
  <model_dir>/
      config.json          {"model_class":"CTRGCN","model_kwargs":{...},"T_MAX":60,"NUM_CLASSES":10}
      model_weights.pt     {"model_state": <state_dict>}
      class_names.json     ["hello","thank you", ...]
      train_mean.npy       shape (288,)
      train_std.npy        shape (288,)
      holistic_landmarker.task   (downloaded automatically if missing)

Config keys used from config.py:
  CTRGCN_MODEL_PATH   — path to the MODEL DIRECTORY (not a .pt file)
  CTRGCN_CLASSES_PATH — ignored (class_names.json is read from model dir)
  CTRGCN_NUM_FRAMES   — sliding window size (overridden by config.json T_MAX)
  GROQ_API_KEY        — optional, for sentence building
"""

import json
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# ─── Keypoint constants (must match webcam_predict.py exactly) ────────────────
POSE_DIM = 132   # 33 landmarks × 4 (x, y, z, visibility)
HAND_DIM = 63    # 21 landmarks × 3 (x, y, z)
FACE_DIM = 30    # 10 sparse landmarks × 3 (x, y, z)
N_FEAT   = 288   # total feature vector per frame

FACE_IDX = [0, 17, 61, 291, 199, 4, 263, 33, 133, 362]   # sparse face landmarks

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)


# ─── Model definitions (identical to webcam_predict.py) ──────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class STGCN(nn.Module):
    """Spatial-Temporal GCN. Input: (B, T, F)"""
    def __init__(self, feat_dim=288, num_classes=10, dropout=0.3):
        super().__init__()
        self.spatial_fc  = nn.Linear(feat_dim, 256)
        self.temporal_bn = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ClassificationHead(256, num_classes, dropout)

    def forward(self, x):
        x = self.spatial_fc(x)
        x = x.transpose(1, 2)
        x = self.temporal_bn(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class CNN1DSignClassifier(nn.Module):
    """1D Temporal CNN. Input: (B, T, F)"""
    def __init__(self, feat_dim=288, num_classes=10, dropout=0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            self._block(feat_dim, 256, 5),
            self._block(256,      256, 5),
            self._block(256,      512, 3),
            self._block(512,      512, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ClassificationHead(512, num_classes, dropout)

    @staticmethod
    def _block(in_ch, out_ch, ks):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class CTRGCN(nn.Module):
    """CTR-GCN. Input: (B, T, F)"""
    def __init__(self, feat_dim=288, num_classes=10, dropout=0.3):
        super().__init__()
        self.joint_embed   = nn.Linear(feat_dim, 256)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ClassificationHead(256, num_classes, dropout)

    def forward(self, x):
        x = self.joint_embed(x)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class SPOTER(nn.Module):
    """Sparse Pose Transformer. Input: (B, T, F)"""
    def __init__(self, feat_dim=288, d_model=108, nhead=9,
                 num_encoder_layers=1, dropout=0.1, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head    = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):
        B  = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(x)
        return self.head(x[:, 0, :])


class SLGTformer(nn.Module):
    """Sign Language Graph Transformer. Input: (B, T, F)"""
    def __init__(self, feat_dim=288, d_model=256, nhead=8, num_layers=4,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.graph_proj = nn.Linear(feat_dim, d_model)
        self.pos_enc    = nn.Embedding(151, d_model)   # T_MAX + 1
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = self.graph_proj(x) + self.pos_enc(pos)
        x   = self.encoder(x)
        x   = x.transpose(1, 2)
        x   = self.pool(x).squeeze(-1)
        return self.head(x)


MODEL_REGISTRY = {
    "STGCN":               STGCN,
    "CNN1DSignClassifier":  CNN1DSignClassifier,
    "CTRGCN":              CTRGCN,
    "SPOTER":              SPOTER,
    "SLGTformer":          SLGTformer,
}


# ─── Holistic landmarker download helper ──────────────────────────────────────

def _ensure_holistic_task(model_dir: str) -> str:
    """Return path to holistic_landmarker.task, downloading if missing."""
    task_path = os.path.join(model_dir, "holistic_landmarker.task")
    if not os.path.exists(task_path):
        # Try the backend root as well
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "holistic_landmarker.task")
        if os.path.exists(alt):
            return alt
        print(f"[SignService] Downloading holistic landmarker (~13 MB) …")
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, task_path)
        print(f"[SignService] Saved to {task_path}")
    return task_path


# ─── Model loader (mirrors webcam_predict.py load_model) ─────────────────────

def _load_model(model_dir: str):
    """
    Load config, weights, class names, and normalisation arrays from model_dir.
    Returns (model, cfg, class_names, mean, std).
    """
    cfg_path = os.path.join(model_dir, "config.json")
    wt_path  = os.path.join(model_dir, "model_weights.pt")
    cn_path  = os.path.join(model_dir, "class_names.json")
    mn_path  = os.path.join(model_dir, "train_mean.npy")
    ms_path  = os.path.join(model_dir, "train_std.npy")

    for p in [cfg_path, wt_path, cn_path, mn_path, ms_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[SignService] Missing required file: {p}\n"
                f"  Expected files: config.json, model_weights.pt, "
                f"class_names.json, train_mean.npy, train_std.npy"
            )

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(cn_path) as f:
        class_names = json.load(f)

    mean = np.load(mn_path)
    std  = np.load(ms_path)
    std  = np.where(std < 1e-6, 1.0, std)

    model_cls = MODEL_REGISTRY[cfg["model_class"]]
    model     = model_cls(**cfg["model_kwargs"])
    ckpt      = torch.load(wt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, cfg, class_names, mean, std


# ─── Skeleton extractor (uses Tasks API, same as webcam_predict.py) ───────────

class _SkeletonExtractor:
    """
    Extracts 288-dim vectors using the legacy MediaPipe Holistic API.

    This matches the extraction style in backend/models/sign/ctr-gcn/predict.py
    and avoids the Windows packet crash seen with Tasks HolisticLandmarker.
    """

    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Extract a 288-dim feature vector from one BGR frame.
        Returns a zero vector (288,) if no landmarks detected (never returns None).
        """
        frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(frame_rgb)

        # Pose: 33 landmarks × 4 (x, y, z, visibility)
        if result.pose_landmarks:
            lms  = result.pose_landmarks.landmark
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                dtype=np.float32,
            ).flatten()
        else:
            pose = np.zeros(POSE_DIM, np.float32)

        # Left hand: 21 landmarks × 3 (x, y, z)
        if result.left_hand_landmarks:
            lms = result.left_hand_landmarks.landmark
            lh  = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                           np.float32).flatten()
        else:
            lh = np.zeros(HAND_DIM, np.float32)

        # Right hand: 21 landmarks × 3 (x, y, z)
        if result.right_hand_landmarks:
            lms = result.right_hand_landmarks.landmark
            rh  = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                           np.float32).flatten()
        else:
            rh = np.zeros(HAND_DIM, np.float32)

        # Sparse face: 10 landmarks × 3 (x, y, z)
        if result.face_landmarks:
            lms  = result.face_landmarks.landmark
            face = np.array(
                [[lms[i].x, lms[i].y, lms[i].z] for i in FACE_IDX],
                dtype=np.float32,
            ).flatten()
        else:
            face = np.zeros(FACE_DIM, np.float32)

        return np.concatenate([pose, lh, rh, face])   # (288,)

    def close(self):
        self._holistic.close()


# ─── Groq sentence builder ────────────────────────────────────────────────────

class _GroqSentenceBuilder:
    _SYSTEM = (
        "You are an ASL-to-English translator. "
        "Given a list of recognised ASL signs, produce a natural English sentence. "
        "ASL omits articles and copula. Be concise. "
        'Respond ONLY with JSON: {"sentence":"...","confidence":0.0–1.0}'
    )

    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def build(self, signs: list) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": f"Signs: {json.dumps(signs)}"},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {
                "sentence":   " ".join(signs).lower().capitalize() + ".",
                "confidence": 0.3,
                "note":       str(e),
            }


# ─── Main SignService ─────────────────────────────────────────────────────────

class SignService:
    """
    Public interface (sign_endpoint WebSocket; client should send up to t_max frames):
      interpret_word(frames: list[bytes])   → dict
      interpret_letter(frames: list[bytes]) → dict
      build_sentence(signs: list[str])      → dict

    Frames are JPEG bytes sent from the Flutter camera.

    IMPORTANT — model_path must point to the MODEL DIRECTORY (the folder
    containing config.json, model_weights.pt, class_names.json, etc.),
    NOT to a single .pt file.  config.py should be updated accordingly:

        CTRGCN_MODEL_PATH = os.path.join(_HERE, "models", "sign", "ctr-gcn")
    """

    def __init__(
        self,
        model_dir: str,
        num_frames: int = 60,
        groq_api_key: str = "",
    ):
        self._lock = threading.Lock()

        print(f"[SignService] Loading model from directory: {model_dir}")
        self._model, self._cfg, self._classes, self._mean, self._std = \
            _load_model(model_dir)

        self._t_max      = self._cfg.get("T_MAX", num_frames)
        self._num_frames = self._t_max
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._model      = self._model.to(self._device)

        print(f"[SignService] Model: {self._cfg['model_class']} | "
              f"Classes: {len(self._classes)} | T_MAX: {self._t_max} | "
              f"Device: {self._device}")

        # Build extractor lazily on the infer thread.
        self._extractor: Optional[_SkeletonExtractor] = None
        print("[SignService] Holistic extractor ready — binds on first infer thread")

        # Optional Groq sentence builder
        self._groq: Optional[_GroqSentenceBuilder] = None
        if groq_api_key:
            try:
                self._groq = _GroqSentenceBuilder(groq_api_key)
                print("[SignService] Groq sentence builder ready")
            except Exception as e:
                print(f"[SignService] Groq init failed (sentence building disabled): {e}")

    @property
    def t_max(self) -> int:
        """Temporal length from model `config.json` (`T_MAX`); matches padding in inference."""
        return int(self._t_max)

    # ── Public API ────────────────────────────────────────────────

    def interpret_word(self, frames: list) -> dict:
        """frames: list of JPEG bytes from phone camera."""
        return self._run_model(frames, mode="word")

    def interpret_letter(self, frames: list) -> dict:
        result = self._run_model(frames, mode="letter")
        sign   = result.get("sign", "?")
        letter = sign[0] if sign else "?"
        return {
            "letter":      letter,
            "confidence":  result.get("confidence", 0.0),
            "alternative": (result.get("alternatives") or [None])[0],
            "note":        result.get("note"),
        }

    def build_sentence(self, signs: list) -> dict:
        if self._groq:
            try:
                return self._groq.build(signs)
            except Exception as e:
                print(f"[SignService] Groq sentence error: {e}")
        sentence = " ".join(s.lower() for s in signs).capitalize() + "."
        return {"sentence": sentence, "confidence": 0.4, "note": "fallback"}

    # ── Core inference ────────────────────────────────────────────

    def _run_model(self, jpeg_frames: list, mode: str) -> dict:
        """
        Decode JPEG frames → 288-dim keypoints → pad/normalise →
        model forward (B, T, 288) → top results.
        Thread-safe (locked).
        """
        error_result = self._make_error(mode)

        if not jpeg_frames:
            error_result["note"] = "No frames provided"
            return error_result

        with self._lock:
            if self._extractor is None:
                self._extractor = _SkeletonExtractor()
                print("[SignService] Holistic extractor active (infer thread)")

            frame_vectors = []

            for jpeg in jpeg_frames:
                arr   = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.size == 0 or frame.shape[0] < 4 or frame.shape[1] < 4:
                    continue
                kp = self._extractor.extract(frame)   # (288,)
                frame_vectors.append(kp)

            if not frame_vectors:
                error_result["note"] = "Could not decode any frames"
                return error_result

            # Stack → (T, 288)
            seq = np.stack(frame_vectors, axis=0)

            # Pad or truncate to T_MAX (same as webcam_predict.py)
            T = seq.shape[0]
            if T < self._t_max:
                pad = np.zeros((self._t_max - T, N_FEAT), np.float32)
                seq = np.concatenate([pad, seq], axis=0)
            elif T > self._t_max:
                seq = seq[-self._t_max:]

            # Normalise with train statistics
            seq = (seq - self._mean) / self._std   # (T_MAX, 288)

            # To tensor: (1, T_MAX, 288)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)

            try:
                with torch.no_grad():
                    logits = self._model(x)               # (1, num_classes)
                    import torch.nn.functional as F

                    # ── Penalize over-dominant classes ────────────────
                    # The model was trained on only 10 signs and tends to
                    # collapse onto "about" for most inputs.  We suppress
                    # any class whose raw logit is more than 3× the mean
                    # logit, forcing the model to spread probability mass
                    # across the remaining classes.
                    _SUPPRESSED = {"about"}   # add more names here if needed
                    _logits = logits[0].clone()
                    _mean_l = _logits.mean()
                    for _ci, _cn in enumerate(self._classes):
                        if _cn.lower() in _SUPPRESSED:
                            # Push logit down to mean − 5σ so it's near-zero prob
                            _logits[_ci] = _mean_l - 5.0 * _logits.std()
                    logits = _logits.unsqueeze(0)

                    probs  = F.softmax(logits, dim=1)[0]  # (num_classes,)
            except Exception as e:
                import traceback; traceback.print_exc()
                error_result["note"] = str(e)
                return error_result

        probs_list = probs.tolist()
        idx        = int(probs.argmax().item())
        conf       = float(probs[idx].item())

        # Top-3 alternatives (excluding the top prediction)
        top3 = sorted(range(len(probs_list)),
                      key=lambda i: probs_list[i], reverse=True)[:3]
        alts = [self._classes[i] for i in top3 if i != idx][:2]

        sign    = self._classes[idx].upper()
        english = self._classes[idx]

        note = None
        if conf < 0.4:
            note = f"Low confidence ({conf:.0%}) — try again in better light"

        if mode == "letter":
            return {
                "sign":               sign,
                "english":            english,
                "confidence":         conf,
                "handshape_observed": "",
                "alternatives":       alts,
                "note":               note,
            }
        return {
            "sign":               sign,
            "english":            english,
            "confidence":         conf,
            "handshape_observed": f"{len(frame_vectors)} frame(s) analysed",
            "alternatives":       alts,
            "note":               note,
        }

    @staticmethod
    def _make_error(mode: str) -> dict:
        if mode == "letter":
            return {"letter": "?", "confidence": 0.0,
                    "alternative": None, "note": "error"}
        return {"sign": "UNCLEAR", "english": "", "confidence": 0.0,
                "handshape_observed": "", "alternatives": [], "note": "error"}

    def close(self):
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None


# ─── Singleton ────────────────────────────────────────────────────────────────

_service: Optional[SignService] = None


def init_sign_service(
    model_path: str,      # should now be the MODEL DIRECTORY path
    classes_path: str,    # kept for API compatibility; ignored (read from model_dir)
    num_frames: int = 60,
    groq_api_key: str = "",
):
    """
    Initialise the singleton SignService.

    model_path should point to the MODEL DIRECTORY (the folder containing
    config.json, model_weights.pt, class_names.json, train_mean.npy,
    train_std.npy).

    Update config.py:
        CTRGCN_MODEL_PATH = os.path.join(_HERE, "models", "sign", "ctr-gcn")
        # (the directory, not a .pt file)
    """
    global _service
    if _service:
        return

    _here = Path(__file__).parent.resolve()

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else _here / pp)

    model_dir = _abs(model_path)

    # If model_path points to a .pt file, use its parent directory
    if model_dir.endswith(".pt") or model_dir.endswith(".pth"):
        model_dir = str(Path(model_dir).parent)
        print(f"[SignService] model_path pointed to a .pt file — "
              f"using parent directory: {model_dir}")

    try:
        _service = SignService(
            model_dir=model_dir,
            num_frames=num_frames,
            groq_api_key=groq_api_key,
        )
    except Exception as e:
        print(f"[SignService] Init failed: {e}")
        import traceback; traceback.print_exc()


def get_service() -> Optional[SignService]:
    return _service