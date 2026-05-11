"""
Live webcam sign language prediction.

Usage:
    python webcam_predict.py --model stgcn
    python webcam_predict.py --model 1D-cnn
    python webcam_predict.py --model ctr-gcn
    python webcam_predict.py --model spoter
    python webcam_predict.py --model slgttransformer

Controls:
    SPACE  — capture the current buffer and predict
    R      — reset / clear the frame buffer
    Q      — quit

Requirements:
    pip install "mediapipe>=0.10.30" opencv-python numpy torch

Model file required (downloaded automatically if missing):
    holistic_landmarker.task  (placed in the same directory as this script)
"""

import argparse
import os
import json
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions (all 5 architectures)
# ─────────────────────────────────────────────────────────────────────────────

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
        self.spatial_fc = nn.Linear(feat_dim, 256)
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
        self.joint_embed = nn.Linear(feat_dim, 256)
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
        B = x.size(0)
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
        self.pos_enc    = nn.Embedding(151, d_model)  # T_MAX+1
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
    "STGCN":              STGCN,
    "CNN1DSignClassifier": CNN1DSignClassifier,
    "CTRGCN":             CTRGCN,
    "SPOTER":             SPOTER,
    "SLGTformer":         SLGTformer,
}

# ─────────────────────────────────────────────────────────────────────────────
# Keypoint extraction helpers (mediapipe >= 0.10.30 tasks API)
# ─────────────────────────────────────────────────────────────────────────────

POSE_DIM = 132   # 33 landmarks × 4 (x, y, z, visibility)
HAND_DIM = 63    # 21 landmarks × 3 (x, y, z)
FACE_DIM = 30    # 10 sparse landmarks × 3 (x, y, z)
N_FEAT   = 288
# Sparse face landmark indices (same as training pipeline)
FACE_IDX = [0, 17, 61, 291, 199, 4, 263, 33, 133, 362]

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
HOLISTIC_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "holistic_landmarker.task")


def ensure_holistic_model():
    """Download the holistic landmarker model file if not present."""
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"Downloading holistic landmarker model (~13 MB) ...")
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
        print(f"Saved to: {HOLISTIC_MODEL_PATH}")
    else:
        print(f"Holistic model found: {HOLISTIC_MODEL_PATH}")


def build_holistic_landmarker():
    """Create a HolisticLandmarker in IMAGE running mode."""
    import mediapipe as mp
    BaseOptions             = mp.tasks.BaseOptions
    HolisticLandmarker      = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    RunningMode             = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
    )
    return HolisticLandmarker.create_from_options(options)


def extract_frame_keypoints(frame_rgb: np.ndarray, landmarker) -> np.ndarray:
    """
    Extract 288-dim keypoint vector from a single RGB frame using the
    mediapipe tasks HolisticLandmarker (>= 0.10.30).

    In this API, result.pose_landmarks is a flat list of 33 NormalizedLandmark
    objects (not a list-of-persons). Same for hand and face landmarks.
    """
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = landmarker.detect(mp_image)

    # ── Pose (33 landmarks × 4: x, y, z, visibility) ─────────────────────────
    if result.pose_landmarks:
        lms  = result.pose_landmarks   # flat list of 33 NormalizedLandmark
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_DIM, np.float32)

    # ── Left hand (21 landmarks × 3: x, y, z) ────────────────────────────────
    if result.left_hand_landmarks:
        lms = result.left_hand_landmarks   # flat list of 21
        lh  = np.array([[lm.x, lm.y, lm.z] for lm in lms], np.float32).flatten()
    else:
        lh = np.zeros(HAND_DIM, np.float32)

    # ── Right hand (21 landmarks × 3: x, y, z) ───────────────────────────────
    if result.right_hand_landmarks:
        lms = result.right_hand_landmarks  # flat list of 21
        rh  = np.array([[lm.x, lm.y, lm.z] for lm in lms], np.float32).flatten()
    else:
        rh = np.zeros(HAND_DIM, np.float32)

    # ── Face (sparse 10 landmarks × 3: x, y, z) ──────────────────────────────
    if result.face_landmarks:
        lms  = result.face_landmarks       # flat list of 478 landmarks
        face = np.array(
            [[lms[i].x, lms[i].y, lms[i].z] for i in FACE_IDX],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(FACE_DIM, np.float32)

    return np.concatenate([pose, lh, rh, face])


def pad_or_truncate(seq, t_max):
    T = seq.shape[0]
    if T >= t_max:
        return seq[-t_max:]
    return np.concatenate(
        [np.zeros((t_max - T, seq.shape[1]), np.float32), seq]
    )


def normalise(seq, mean, std):
    return (seq - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    wt_path  = os.path.join(model_dir, "model_weights.pt")
    cn_path  = os.path.join(model_dir, "class_names.json")
    mn_path  = os.path.join(model_dir, "train_mean.npy")
    ms_path  = os.path.join(model_dir, "train_std.npy")

    for p in [cfg_path, wt_path, cn_path, mn_path, ms_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_predictions(frame, predictions, recording, buffer_len, t_max):
    """Overlay prediction results and status on the frame."""
    h, w = frame.shape[:2]

    # Status bar at top
    status_color = (0, 0, 200) if recording else (50, 50, 50)
    cv2.rectangle(frame, (0, 0), (w, 50), status_color, -1)

    if recording:
        status_text = f"RECORDING  {buffer_len}/{t_max} frames"
    else:
        status_text = "Press SPACE to predict  |  R to reset  |  Q to quit"

    cv2.putText(frame, status_text, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Buffer progress bar
    if t_max > 0:
        bar_w = int(w * min(buffer_len / t_max, 1.0))
        cv2.rectangle(frame, (0, 45), (bar_w, 50), (0, 255, 100), -1)

    # Prediction results panel
    if predictions:
        panel_h = 30 + len(predictions) * 38
        cv2.rectangle(frame, (0, h - panel_h - 10), (360, h), (20, 20, 20), -1)
        cv2.putText(frame, "Predictions:", (10, h - panel_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for i, (label, prob) in enumerate(predictions):
            y = h - panel_h + 15 + (i + 1) * 38
            bar_len = int(300 * prob)
            color = (0, 200, 100) if i == 0 else (100, 150, 200)
            cv2.rectangle(frame, (10, y - 18), (10 + bar_len, y + 4), color, -1)
            text = f"{label}  {prob * 100:.1f}%"
            cv2.putText(frame, text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(model_name: str):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    print(f"Loading model from: {model_dir}")
    model, cfg, class_names, mean, std = load_model(model_dir)

    t_max       = cfg["T_MAX"]
    num_classes = cfg["NUM_CLASSES"]
    top_k       = min(5, num_classes)

    print(f"Model: {cfg['model_class']}  |  Classes: {num_classes}  |  T_MAX: {t_max}")
    print(f"Class names: {class_names}")
    print()
    print("Controls:")
    print("  SPACE — predict on current buffer")
    print("  R     — reset buffer")
    print("  Q     — quit")
    print()

    ensure_holistic_model()
    landmarker = build_holistic_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check that a camera is connected.")

    frame_buffer = []   # list of (288,) arrays
    predictions  = []   # list of (label, prob) tuples
    recording    = True # always collecting frames

    print("Webcam started. Press SPACE to predict.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp = extract_frame_keypoints(frame_rgb, landmarker)
        frame_buffer.append(kp)

        # Keep only the last t_max frames
        if len(frame_buffer) > t_max:
            frame_buffer = frame_buffer[-t_max:]

        # Draw overlay
        display = draw_predictions(
            frame.copy(), predictions, recording,
            len(frame_buffer), t_max
        )
        cv2.imshow(f"Sign Language Prediction — {cfg['model_class']}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break

        elif key == ord('r'):
            frame_buffer = []
            predictions  = []
            print("Buffer reset.")

        elif key == ord(' '):
            if len(frame_buffer) == 0:
                print("Buffer is empty. Sign something first.")
                continue

            # Build sequence
            seq = np.stack(frame_buffer, axis=0)   # (T, 288)
            seq = pad_or_truncate(seq, t_max)       # (T_MAX, 288)
            seq = normalise(seq, mean, std)         # (T_MAX, 288)

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T_MAX, 288)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0]
                top    = probs.topk(top_k)

            predictions = [
                (class_names[idx.item()], prob.item())
                for prob, idx in zip(top.values, top.indices)
            ]

            print("\nPrediction:")
            for label, prob in predictions:
                print(f"  {label:25s}  {prob * 100:.1f}%")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam sign language prediction")
    parser.add_argument(
        "--model",
        type=str,
        default="stgcn",
        choices=["stgcn", "1D-cnn", "ctr-gcn", "spoter", "slgttransformer"],
        help="Which model to use for prediction (default: stgcn)",
    )
    args = parser.parse_args()
    run_webcam(args.model)
