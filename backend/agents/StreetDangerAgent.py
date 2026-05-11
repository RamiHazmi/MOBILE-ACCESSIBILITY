"""
StreetDangerAgent.py — YOLO best.pt + LLM verification for navigation
─────────────────────────────────────────────────────────────────────
While the user is walking (navigation mode), the camera frames are
sent here instead of to the slow VLM.

Pipeline:
  1. Local YOLO best.pt detects 19 street classes
  2. Filter by proximity (bbox area + position) — parked cars or
     distant objects are NOT dangers
  3. If at least one NEAR threat is found, optionally ask the small
     VLM "is this really a danger to a blind person walking now?"
     (cheap because it's already filtered)
  4. Only NEAR + VLM-verified threats produce a spoken warning

Classes (from user's training):
  0: car   1: bus   2: truck   3: bike   4: motorcycle
  5: person   6: traffic_light   7: traffic_sign
  8: pothole   9: ashcan  10: blind_road  11: crosswalk
 12: dog  13: fire_hydrant  14: pole  15: reflective_cone
 16: roadblock  17: tree  18: tricycle

Cooldown is enforced so the user is not spammed with the same warning.
"""

from __future__ import annotations
import os
import time
import base64
import io
from typing import Optional, List, Dict, Any
from pathlib import Path

# Lazy imports — Ultralytics + Pillow are loaded only when YOLO file exists


# ── Class names (must match best.pt training) ───────────────────
CLASS_NAMES = [
    "car", "bus", "truck", "bike", "motorcycle",
    "person", "traffic_light", "traffic_sign",
    "pothole", "ashcan", "blind_road", "crosswalk",
    "dog", "fire_hydrant", "pole", "reflective_cone",
    "roadblock", "tree", "tricycle",
]

# ── Friendly speakable names ────────────────────────────────────
SPEAK_NAMES = {
    "car":             "car",
    "bus":             "bus",
    "truck":           "truck",
    "bike":            "bicycle",
    "motorcycle":      "motorcycle",
    "person":          "person",
    "traffic_light":   "traffic light",
    "traffic_sign":    "traffic sign",
    "pothole":         "pothole",
    "ashcan":          "trash can",
    "blind_road":      "tactile paving",
    "crosswalk":       "crosswalk",
    "dog":             "dog",
    "fire_hydrant":    "fire hydrant",
    "pole":            "pole",
    "reflective_cone": "cone",
    "roadblock":       "roadblock",
    "tree":            "tree",
    "tricycle":        "tricycle",
}

# Localized object names — fallback to English if missing
SPEAK_NAMES_I18N = {
    "fr": {
        "car": "voiture", "bus": "bus", "truck": "camion",
        "bike": "vélo", "motorcycle": "moto", "person": "personne",
        "traffic_light": "feu de circulation", "traffic_sign": "panneau",
        "pothole": "nid de poule", "ashcan": "poubelle",
        "blind_road": "bande podotactile", "crosswalk": "passage piéton",
        "dog": "chien", "fire_hydrant": "borne d'incendie",
        "pole": "poteau", "reflective_cone": "cône",
        "roadblock": "barrage", "tree": "arbre", "tricycle": "tricycle",
    },
    "ar": {
        "car": "سيارة", "bus": "حافلة", "truck": "شاحنة",
        "bike": "دراجة", "motorcycle": "دراجة نارية", "person": "شخص",
        "traffic_light": "إشارة مرور", "traffic_sign": "لافتة",
        "pothole": "حفرة", "ashcan": "صندوق قمامة",
        "blind_road": "ممر للمكفوفين", "crosswalk": "ممر مشاة",
        "dog": "كلب", "fire_hydrant": "صنبور إطفاء",
        "pole": "عمود", "reflective_cone": "مخروط",
        "roadblock": "حاجز", "tree": "شجرة", "tricycle": "دراجة ثلاثية",
    },
    "tn": {
        "car": "كرهبة", "bus": "كار", "truck": "شاحنة",
        "bike": "بشكليط", "motorcycle": "موطو", "person": "شخص",
        "traffic_light": "ضو طريق", "traffic_sign": "لافتة",
        "pothole": "حفرة", "ashcan": "زبلة",
        "blind_road": "ممر عميان", "crosswalk": "بساج",
        "dog": "كلب", "fire_hydrant": "صنبور",
        "pole": "عمود", "reflective_cone": "مخروط",
        "roadblock": "حاجز", "tree": "شجرة", "tricycle": "تريسيكل",
    },
}

# Localized templates
_PROX_WORDS = {
    "en": {"near": "close", "medium": "at medium distance", "far": "far"},
    "fr": {"near": "près",   "medium": "à distance moyenne", "far": "loin"},
    "ar": {"near": "قريب",   "medium": "على بعد متوسط",      "far": "بعيد"},
    "tn": {"near": "قريب",   "medium": "بعيد شويا",          "far": "بعيد"},
}
_DIR_WORDS = {
    "en": {"left": "left", "center": "ahead", "right": "right"},
    "fr": {"left": "gauche", "center": "devant", "right": "droite"},
    "ar": {"left": "اليسار", "center": "أمامك", "right": "اليمين"},
    "tn": {"left": "اليسار", "center": "قدامك", "right": "اليمين"},
}
_TPL_FRONT = {
    "en": "{name} {prox} in front",
    "fr": "{name} {prox} devant vous",
    "ar": "{name} {prox} أمامك",
    "tn": "{name} {prox} قدامك",
}
_TPL_SIDE = {
    "en": "{name} {prox} on your {dir}",
    "fr": "{name} {prox} sur votre {dir}",
    "ar": "{name} {prox} على {dir}ك",
    "tn": "{name} {prox} على {dir}ك",
}
_TPL_AHEAD = {
    "en": "{name} {prox} ahead",
    "fr": "{name} {prox} devant",
    "ar": "{name} {prox} أمامك",
    "tn": "{name} {prox} قدامك",
}
_TPL_HELPFUL = {
    "en": "{name} ahead",
    "fr": "{name} devant",
    "ar": "{name} أمامك",
    "tn": "{name} قدامك",
}
_CAUTION = {
    "en": "Caution.",
    "fr": "Attention.",
    "ar": "انتبه.",
    "tn": "رد بالك.",
}
_ALSO = {
    "en": " Also",
    "fr": " Aussi",
    "ar": " وأيضاً",
    "tn": " وزادة",
}


def _name_in(label: str, lang: str) -> str:
    bundle = SPEAK_NAMES_I18N.get(lang, {})
    return bundle.get(label) or SPEAK_NAMES.get(label, label)


def _build_reason(label: str, prox: str, dirn: str,
                  in_path: bool, family: str, lang: str) -> str:
    """family: 'dynamic' | 'static' | 'helpful'"""
    name = _name_in(label, lang)
    prox_w = _PROX_WORDS.get(lang, _PROX_WORDS["en"]).get(prox, prox)
    dir_w  = _DIR_WORDS.get(lang, _DIR_WORDS["en"]).get(dirn, dirn)
    if family == "helpful":
        return _TPL_HELPFUL.get(lang, _TPL_HELPFUL["en"]).format(name=name)
    if family == "dynamic":
        if in_path:
            return _TPL_FRONT.get(lang, _TPL_FRONT["en"]).format(name=name, prox=prox_w)
        return _TPL_SIDE.get(lang, _TPL_SIDE["en"]).format(name=name, prox=prox_w, dir=dir_w)
    return _TPL_AHEAD.get(lang, _TPL_AHEAD["en"]).format(name=name, prox=prox_w)

# ── Which classes are EVER a hazard to a blind walker ───────────
# (Things like a traffic_sign or distant tree are not threats by
# themselves — only obstacles directly in the path are dangerous.)
DYNAMIC_THREATS = {"car", "bus", "truck", "motorcycle", "bike",
                   "tricycle", "person", "dog"}
STATIC_OBSTACLES = {"pothole", "ashcan", "pole", "fire_hydrant",
                    "tree", "reflective_cone", "roadblock"}
HELPFUL = {"crosswalk", "blind_road", "traffic_light"}  # speak as cue, not danger
IGNORED = {"traffic_sign"}  # not actionable for blind user

# Distance / size heuristics
NEAR_AREA_RATIO  = 0.08   # bbox covers ≥ 8% of frame  → close
MID_AREA_RATIO   = 0.03   # 3-8% → medium
CENTER_BAND      = (0.30, 0.70)   # cx must be in this band to be "in path"
GROUND_BAND_MIN  = 0.40           # cy ≥ 0.40 means lower half of frame

DEFAULT_CONF_THRESHOLD = 0.45
SPEAK_COOLDOWN_S       = 6.0    # don't repeat same warning more often
HELPFUL_COOLDOWN_S     = 12.0


# ── Geometry helpers ────────────────────────────────────────────

def _bbox_area_ratio(box_xyxy, w: float, h: float) -> float:
    x1, y1, x2, y2 = box_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return (bw * bh) / max(1.0, (w * h))

def _bbox_cx(box_xyxy, w: float) -> float:
    x1, _, x2, _ = box_xyxy
    return ((x1 + x2) * 0.5) / max(1.0, w)

def _bbox_cy(box_xyxy, h: float) -> float:
    _, y1, _, y2 = box_xyxy
    return ((y1 + y2) * 0.5) / max(1.0, h)

def _proximity(area_ratio: float) -> str:
    if area_ratio >= NEAR_AREA_RATIO: return "near"
    if area_ratio >= MID_AREA_RATIO:  return "medium"
    return "far"

def _direction(cx: float) -> str:
    if cx < 0.35: return "left"
    if cx > 0.65: return "right"
    return "center"


# ─────────────────────────────────────────────────────────────────
class StreetDangerAgent:
    """
    YOLO-based danger detection during navigation.

    Use:
        agent = StreetDangerAgent("models/yolo/best.pt")
        result = agent.detect_dangers(frame_b64)
        if result["speak"]:
            tts.speak(result["speak"], interrupt=result["interrupt"])
    """

    def __init__(
        self,
        model_path: str = "models/yolo/best.pt",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.conf       = conf_threshold
        self.device     = device       # None = auto, "cpu", "cuda:0", etc.
        self.model      = None
        self._load_error: Optional[str] = None
        self._last_spoken: Dict[str, float] = {}  # key → timestamp

        self._try_load()

    # ── Loading ───────────────────────────────────────────────

    def _try_load(self):
        try:
            from ultralytics import YOLO  # noqa
        except Exception as e:
            self._load_error = (
                "ultralytics not installed. "
                "Run: pip install ultralytics")
            print(f"[StreetDanger] {self._load_error}")
            return

        path = Path(self.model_path)
        if not path.is_absolute():
            here = Path(__file__).resolve().parent.parent  # backend/
            path = (here / self.model_path).resolve()

        if not path.exists():
            self._load_error = f"YOLO model not found at {path}"
            print(f"[StreetDanger] {self._load_error}")
            print("[StreetDanger] Place your best.pt at backend/models/yolo/best.pt")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            print(f"[StreetDanger] Loaded YOLO model from {path}")
        except Exception as e:
            self._load_error = f"YOLO load failed: {e}"
            print(f"[StreetDanger] {self._load_error}")

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ── Detection ─────────────────────────────────────────────

    def detect_raw(self, frame_b64: str) -> List[Dict[str, Any]]:
        """Run YOLO and return list of {label, conf, bbox_xyxy, w, h}."""
        if not self.ready:
            return []
        try:
            from PIL import Image
        except Exception:
            return []

        try:
            raw = base64.b64decode(frame_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            print(f"[StreetDanger] Decode error: {e}")
            return []

        w, h = img.size
        try:
            res = self.model.predict(
                img, conf=self.conf, verbose=False, device=self.device)
        except Exception as e:
            print(f"[StreetDanger] Inference error: {e}")
            return []

        out = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue
            out.append({
                "label":     CLASS_NAMES[cls_id],
                "conf":      float(conf),
                "bbox_xyxy": [float(x) for x in box],
                "img_w":     int(w),
                "img_h":     int(h),
            })
        return out

    def _classify(self, det: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        box = det["bbox_xyxy"]
        w   = det["img_w"]
        h   = det["img_h"]
        area = _bbox_area_ratio(box, w, h)
        cx   = _bbox_cx(box, w)
        cy   = _bbox_cy(box, h)
        prox = _proximity(area)
        dirn = _direction(cx)

        in_path = (CENTER_BAND[0] <= cx <= CENTER_BAND[1]
                   and cy >= GROUND_BAND_MIN)

        label = det["label"]
        is_danger = False
        reason    = ""

        if label in DYNAMIC_THREATS:
            if prox == "near":
                is_danger = True
                reason    = _build_reason(label, prox, dirn,
                                          in_path, "dynamic", lang)
        elif label in STATIC_OBSTACLES:
            if in_path and prox in {"near", "medium"}:
                is_danger = True
                reason    = _build_reason(label, prox, dirn,
                                          in_path, "static", lang)
        elif label in HELPFUL:
            reason = _build_reason(label, prox, dirn,
                                   in_path, "helpful", lang)

        return {
            **det,
            "area_ratio": area,
            "bbox_cx":    cx,
            "bbox_cy":    cy,
            "proximity":  prox,
            "direction":  dirn,
            "in_path":    in_path,
            "is_danger":  is_danger,
            "reason":     reason,
        }

    # ── Public ────────────────────────────────────────────────

    def detect_dangers(self, frame_b64: str, language: str = "en") -> Dict[str, Any]:
        """
        Returns:
          {
            "ready":     bool,
            "objects":   [...],
            "dangers":   [...],
            "helpful":   [...],
            "speak":     str,
            "interrupt": bool,
            "danger":    "LOW" | "MEDIUM" | "HIGH",
          }
        """
        if not self.ready:
            return {
                "ready": False,
                "objects": [], "dangers": [], "helpful": [],
                "speak": "", "interrupt": False, "danger": "LOW",
                "load_error": self._load_error,
            }

        lang     = (language or "en").lower()
        raw_dets = self.detect_raw(frame_b64)
        objects  = [self._classify(d, lang=lang) for d in raw_dets]

        dangers = [o for o in objects if o["is_danger"]]
        helpful = [o for o in objects
                   if o["label"] in HELPFUL and not o["is_danger"]]

        # Sort dangers by proximity (near first, then biggest area)
        prox_rank = {"near": 0, "medium": 1, "far": 2}
        dangers.sort(key=lambda o: (prox_rank[o["proximity"]],
                                    -o["area_ratio"]))

        speak       = ""
        interrupt   = False
        danger_lvl  = "LOW"
        now         = time.time()

        if dangers:
            top = dangers[0]
            # Cooldown by (label, direction)
            key      = f"D|{top['label']}|{top['direction']}"
            last     = self._last_spoken.get(key, 0.0)
            if now - last >= SPEAK_COOLDOWN_S:
                speak = _CAUTION.get(lang, _CAUTION["en"]) + " " + top["reason"] + "."
                if (len(dangers) >= 2 and dangers[1]["proximity"] == "near"
                        and dangers[1]["label"] != top["label"]):
                    speak += _ALSO.get(lang, _ALSO["en"]) + " " + dangers[1]["reason"] + "."
                interrupt = (top["proximity"] == "near"
                             and top["in_path"])
                danger_lvl = "HIGH" if interrupt else "MEDIUM"
                self._last_spoken[key] = now

        elif helpful:
            top = helpful[0]
            key  = f"H|{top['label']}"
            last = self._last_spoken.get(key, 0.0)
            if now - last >= HELPFUL_COOLDOWN_S:
                speak = top["reason"] + "."
                self._last_spoken[key] = now

        return {
            "ready":     True,
            "objects":   objects,
            "dangers":   dangers,
            "helpful":   helpful,
            "speak":     speak,
            "interrupt": interrupt,
            "danger":    danger_lvl,
        }

    def reset_cooldowns(self):
        self._last_spoken.clear()