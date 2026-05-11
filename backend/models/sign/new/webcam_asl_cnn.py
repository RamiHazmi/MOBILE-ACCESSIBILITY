"""
ASL Alphabet Recognition — CNN on Sign MNIST
=============================================
Uses a CNN trained on Sign Language MNIST (28x28 grayscale images).
Detects hand region via MediaPipe, crops and resizes to 28x28,
then classifies with the CNN.

Recognizes 24 static ASL letters: A-Y (excluding J and Z which need motion).

Usage:
    python asl-alphabet/webcam_asl_cnn.py

Controls:
    Q / ESC — quit
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "asl_cnn.pt")

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
NUM_CLASSES = 24

# ── CNN Model (must match train_asl_cnn.py) ───────────────────────────────────
class ASL_CNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = ASL_CNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_hand_roi(roi_bgr):
    """
    Convert a hand ROI (BGR) to the 28x28 grayscale tensor the CNN expects.
    Mimics Sign MNIST preprocessing: grayscale, resize to 28x28, normalize.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor, resized


def run_webcam():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Run: python asl-alphabet/train_asl_cnn.py")
        sys.exit(1)

    print("Loading CNN model...")
    model = load_model()
    print("Model loaded.")

    # MediaPipe hand landmarker for hand detection/bounding box
    HAND_MODEL_PATH = os.path.join(
        os.path.dirname(SCRIPT_DIR), "holistic_landmarker.task"
    )

    # Use MediaPipe Hands (Tasks API) for hand bounding box
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    # Download hand landmarker model if needed
    HAND_TASK_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "hand_landmarker.task")
    if not os.path.exists(HAND_TASK_PATH):
        # Try to use holistic model path
        HAND_TASK_PATH = HAND_MODEL_PATH

    # Fall back to mp.solutions.hands if task file not available
    use_tasks_api = os.path.exists(HAND_TASK_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pred_buffer = collections.deque(maxlen=10)
    conf_buffer = collections.deque(maxlen=10)

    # ROI box: user can adjust with keyboard
    # Default: center square of frame
    ROI_SIZE = 200  # pixels

    print("\nASL Alphabet Recognition (CNN)")
    print("Show your hand inside the GREEN BOX")
    print("Press Q to quit\n")

    # We'll use a fixed ROI box approach (most reliable for Sign MNIST)
    # The user positions their hand in the box
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        # Define ROI box (center of frame)
        cx, cy = w // 2, h // 2
        x1 = cx - ROI_SIZE // 2
        y1 = cy - ROI_SIZE // 2
        x2 = cx + ROI_SIZE // 2
        y2 = cy + ROI_SIZE // 2

        # Clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        predicted_letter = ""
        confidence = 0.0
        preview_28 = None

        if roi.size > 0:
            tensor, preview_28 = preprocess_hand_roi(roi)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                top_idx = probs.argmax().item()
                confidence = probs[top_idx].item()
                predicted_letter = LETTERS[top_idx]
                pred_buffer.append(predicted_letter)
                conf_buffer.append(confidence)

        # Smoothed prediction (majority vote)
        smooth_pred = ""
        smooth_conf = 0.0
        if pred_buffer:
            counter = collections.Counter(pred_buffer)
            smooth_pred, count = counter.most_common(1)[0]
            # Average confidence for the winning letter
            matching_confs = [c for p, c in zip(pred_buffer, conf_buffer) if p == smooth_pred]
            smooth_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0

        # ── Draw UI ──────────────────────────────────────────────────────────

        # ROI box
        box_color = (0, 255, 0) if smooth_conf > 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "ASL Alphabet Recognition (A-Y)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Prediction display
        if smooth_pred:
            # Big letter on right
            cv2.putText(frame, smooth_pred, (w - 120, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 100), 8)
            # Confidence bar
            bar_w = int(300 * smooth_conf)
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 65), (0, 200, 100), -1)
            cv2.putText(frame, f"{smooth_pred}  {smooth_conf*100:.0f}%",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Waiting...", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)

        # Show top-3 predictions
        if roi.size > 0:
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                top3 = probs.topk(3)
            panel_y = h - 110
            cv2.rectangle(frame, (0, panel_y - 5), (250, h), (20, 20, 20), -1)
            cv2.putText(frame, "Top predictions:", (10, panel_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
                letter = LETTERS[idx.item()]
                p = prob.item()
                bar = int(200 * p)
                color = (0, 200, 100) if i == 0 else (100, 150, 200)
                y_pos = panel_y + 35 + i * 28
                cv2.rectangle(frame, (10, y_pos - 14), (10 + bar, y_pos + 4), color, -1)
                cv2.putText(frame, f"{letter}: {p*100:.0f}%", (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Show 28x28 preview (what the CNN sees)
        if preview_28 is not None:
            preview_big = cv2.resize(preview_28, (84, 84), interpolation=cv2.INTER_NEAREST)
            preview_bgr = cv2.cvtColor(preview_big, cv2.COLOR_GRAY2BGR)
            frame[y1:y1+84, x2+5:x2+89] = preview_bgr
            cv2.putText(frame, "CNN input", (x2 + 5, y1 + 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.putText(frame, "Q: quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow("ASL Alphabet Recognition (CNN)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run_webcam()
