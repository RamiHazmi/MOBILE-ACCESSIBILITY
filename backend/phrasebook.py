"""
phrasebook.py — multilingual templates for everything Claude speaks.

The user's chosen language flows from DialogueAgent → state["language"]
all the way to action_speak.  Every spoken sentence in the orchestrator
should be built via t() so the user hears their own language.

Supported language codes: en, fr, ar, tn (Tunisian Derja).
"""

from __future__ import annotations
from typing import Optional, Dict, Any


# ── Helpers ─────────────────────────────────────────────────────

def t(key: str, lang: str = "en", **kwargs) -> str:
    """
    Look up a phrase template and format it with kwargs.

      t("found_target", lang="ar", target="هاتف",
        turn="...", dist="قريباً")
    """
    lang = (lang or "en").lower()
    bundle = _PHRASES.get(key)
    if not bundle:
        return ""
    template = bundle.get(lang) or bundle.get("en") or ""
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def turn_phrase(position: str, lang: str = "en") -> str:
    """'left' / 'center' / 'right' → spoken phrase in the user's language."""
    bundle = _TURNS.get((position or "center").lower(), _TURNS["center"])
    return bundle.get(lang) or bundle.get("en") or ""


def distance_phrase(distance: str, lang: str = "en") -> str:
    """'close' / 'medium' / 'far' / 'nearby' → localized adjective."""
    bundle = _DISTANCES.get((distance or "nearby").lower(), _DISTANCES["nearby"])
    return bundle.get(lang) or bundle.get("en") or distance


def position_word(position: str, lang: str = "en") -> str:
    """Bare 'left'/'center'/'right' for compact warnings."""
    bundle = _POSITIONS.get((position or "center").lower(), _POSITIONS["center"])
    return bundle.get(lang) or bundle.get("en") or position


# ── Phrase data ─────────────────────────────────────────────────

_PHRASES: Dict[str, Dict[str, str]] = {
    # --- Find-object node -------------------------------------
    "found_target": {
        "en": "Found your {target}. {turn}, {dist} away.",
        "fr": "J'ai trouvé votre {target}. {turn}, à {dist}.",
        "ar": "وجدت {target}. {turn}، على بعد {dist}.",
        "tn": "لقيت {target} متاعك. {turn}، {dist}.",
    },
    "searching_camera_unclear": {
        "en": "Camera view is unclear. Make sure the lens is uncovered and there is light.",
        "fr": "La vue de la caméra n'est pas claire. Vérifiez que l'objectif est dégagé.",
        "ar": "الكاميرا غير واضحة. تأكد من أن العدسة مكشوفة وهناك إضاءة كافية.",
        "tn": "الكاميرا موش واضحة. شوف العدسة مكشوفة و في ضو.",
    },
    "cannot_see_with_others": {
        "en": "I cannot see your {target} yet. I can see {names}. Try turning slowly to your right.",
        "fr": "Je ne vois pas encore votre {target}. Je vois {names}. Tournez lentement vers la droite.",
        "ar": "لا أرى {target} بعد. أرى {names}. حاول أن تستدير ببطء إلى اليمين.",
        "tn": "ما نشوفش {target} متاعك. نشوف {names}. دور بشويا على اليمين.",
    },
    "cannot_see_alone": {
        "en": "I cannot see your {target}. Try turning around slowly or moving to a brighter area.",
        "fr": "Je ne vois pas votre {target}. Tournez doucement ou allez vers un endroit plus éclairé.",
        "ar": "لا أرى {target}. حاول أن تستدير ببطء أو انتقل إلى مكان أكثر إضاءة.",
        "tn": "ما نشوفش {target}. دور بشويا ولا روح لبلاصة فيها ضو أكثر.",
    },
    "describe_outdoor_indoor": {
        "en": "You are {env}, in a {scene_type}. I can see {objects}. {guidance}",
        "fr": "Vous êtes {env}, dans un {scene_type}. Je vois {objects}. {guidance}",
        "ar": "أنت {env}، في {scene_type}. أرى {objects}. {guidance}",
        "tn": "انت {env}، في {scene_type}. نشوف {objects}. {guidance}",
    },
    "describe_simple": {
        "en": "I can see {objects}. {guidance}",
        "fr": "Je vois {objects}. {guidance}",
        "ar": "أرى {objects}. {guidance}",
        "tn": "نشوف {objects}. {guidance}",
    },
    "describe_unclear": {
        "en": "I cannot see clearly. Please point the camera forward.",
        "fr": "Je ne vois pas clairement. Pointez la caméra vers l'avant.",
        "ar": "لا أرى بوضوح. وجّه الكاميرا إلى الأمام من فضلك.",
        "tn": "ما نشوفش مليح. وجّه الكاميرا قدامك.",
    },
    "describe_nothing_notable": {
        "en": "nothing notable",
        "fr": "rien de particulier",
        "ar": "لا شيء مهم",
        "tn": "حتى حاجة مهمة",
    },
    "object_phrase": {
        # Used inside describe lists: "a phone on your left, close away"
        "en": "a {label} on your {position}, {dist} away",
        "fr": "un {label} sur votre {position}, à {dist}",
        "ar": "{label} على {position}ك، على بعد {dist}",
        "tn": "{label} على {position}ك، {dist}",
    },
    "opening_camera_for_find": {
        "en": "Opening camera to find your {target}. Hold the phone in front of you and slowly turn around.",
        "fr": "J'ouvre la caméra pour trouver votre {target}. Tenez le téléphone devant vous et tournez lentement.",
        "ar": "أفتح الكاميرا للبحث عن {target}. أمسك الهاتف أمامك واستدر ببطء.",
        "tn": "نحلّ الكاميرا باش نلقى {target}. شد التيليفون قدامك ودور بشويا.",
    },

    # --- Emotion phrases (DESCRIBE mode only) ----------------
    # Phrased cautiously: "looks happy" rather than "is happy" —
    # emotion classification is approximate and we don't want to
    # assert a stranger's actual feelings.
    "emotion_face_phrase": {
        # For center position the orchestrator passes a "front" word,
        # so phrasing reads "in front of you looks happy".  Left/right
        # use the possessive "on your X".
        "en": "a person {position} looks {emotion}",
        "fr": "une personne {position} a l'air {emotion}",
        "ar": "شخص {position} يبدو {emotion}",
        "tn": "واحد {position} يبان {emotion}",
    },
    "face_position_left": {
        "en": "on your left",  "fr": "sur votre gauche",
        "ar": "على يسارك",      "tn": "على يسارك",
    },
    "face_position_right": {
        "en": "on your right", "fr": "sur votre droite",
        "ar": "على يمينك",      "tn": "على يمينك",
    },
    "face_position_center": {
        "en": "in front of you", "fr": "devant vous",
        "ar": "أمامك",            "tn": "قدامك",
    },
    "emotion_happy": {
        "en": "happy",        "fr": "heureux",
        "ar": "سعيد",          "tn": "فرحان",
    },
    "emotion_sad": {
        "en": "sad",          "fr": "triste",
        "ar": "حزين",          "tn": "محزون",
    },
    "emotion_angry": {
        "en": "angry",        "fr": "en colère",
        "ar": "غاضب",          "tn": "متعصب",
    },
    "emotion_fear": {
        "en": "afraid",       "fr": "effrayé",
        "ar": "خائف",          "tn": "خايف",
    },
    "emotion_surprise": {
        "en": "surprised",    "fr": "surpris",
        "ar": "متفاجئ",        "tn": "مستغرب",
    },
    "emotion_disgust": {
        "en": "displeased",   "fr": "dégoûté",
        "ar": "مشمئز",          "tn": "قارف",
    },
    "emotion_neutral": {
        "en": "calm",         "fr": "calme",
        "ar": "هادئ",          "tn": "هادي",
    },

    # --- Navigation -------------------------------------------
    "nav_found_route": {
        "en": "Found {dest}. It is {dist} away, about {dur} {mode_word}. Starting now. {first}. In {first_d}.",
        "fr": "Trouvé {dest}. C'est à {dist}, environ {dur} {mode_word}. On commence. {first}. Dans {first_d}.",
        "ar": "وجدت {dest}. على بعد {dist}، حوالي {dur} {mode_word}. نبدأ الآن. {first}. خلال {first_d}.",
        "tn": "لقيت {dest}. على بعد {dist}، تقريباً {dur} {mode_word}. نبداو توا. {first}. في {first_d}.",
    },
    "nav_arrived": {
        "en": "You have arrived at {dest}.",
        "fr": "Vous êtes arrivé à {dest}.",
        "ar": "لقد وصلت إلى {dest}.",
        "tn": "وصلت لـ {dest}.",
    },
    "nav_continue_far": {
        "en": "Continue for {dist}.",
        "fr": "Continuez sur {dist}.",
        "ar": "استمر لمسافة {dist}.",
        "tn": "كمل على {dist}.",
    },
    "nav_in_dist_then": {
        "en": "In {dist}, {instruction}.",
        "fr": "Dans {dist}, {instruction}.",
        "ar": "خلال {dist}، {instruction}.",
        "tn": "في {dist}، {instruction}.",
    },
    "nav_now": {
        "en": "{instruction} now.",
        "fr": "{instruction} maintenant.",
        "ar": "{instruction} الآن.",
        "tn": "{instruction} توا.",
    },
    "nav_advance_step": {
        "en": "{instruction}. In {dist}.",
        "fr": "{instruction}. Dans {dist}.",
        "ar": "{instruction}. خلال {dist}.",
        "tn": "{instruction}. في {dist}.",
    },
    "nav_off_route": {
        "en": "You appear to have left the route. Say recalculate to get a new path.",
        "fr": "Vous semblez avoir quitté l'itinéraire. Dites recalculer pour un nouveau chemin.",
        "ar": "يبدو أنك خرجت عن المسار. قل إعادة حساب للحصول على مسار جديد.",
        "tn": "خرجت من الطريق. قول إعادة حساب باش نعطيك طريق جديد.",
    },
    "nav_mode_walk": {
        "en": "walking",   "fr": "à pied", "ar": "مشياً", "tn": "بالرجل",
    },
    "nav_mode_drive": {
        "en": "by car",    "fr": "en voiture", "ar": "بالسيارة", "tn": "بالكرهبة",
    },
    "nav_turn_left": {
        "en": "Turn left",  "fr": "Tournez à gauche",
        "ar": "انعطف يساراً", "tn": "لف لليسار",
    },
    "nav_turn_right": {
        "en": "Turn right", "fr": "Tournez à droite",
        "ar": "انعطف يميناً", "tn": "لف لليمين",
    },
    "nav_slight_left": {
        "en": "Bear slightly left", "fr": "Légèrement à gauche",
        "ar": "مل قليلاً يساراً",   "tn": "مل شويا لليسار",
    },
    "nav_slight_right": {
        "en": "Bear slightly right", "fr": "Légèrement à droite",
        "ar": "مل قليلاً يميناً",     "tn": "مل شويا لليمين",
    },
    "nav_straight": {
        "en": "Continue straight", "fr": "Continuez tout droit",
        "ar": "تابع مستقيماً",      "tn": "كمل ديركت",
    },
    "nav_uturn": {
        "en": "Make a U-turn", "fr": "Faites demi-tour",
        "ar": "قم بالدوران للخلف", "tn": "دور للور",
    },
    "nav_arrive": {
        "en": "You will arrive", "fr": "Vous allez arriver",
        "ar": "ستصل",              "tn": "تو توصل",
    },
    "nav_on_street": {
        "en": " onto {street}",
        "fr": " sur {street}",
        "ar": " إلى {street}",
        "tn": " على {street}",
    },

    # --- Memory Mode ------------------------------------------
    "memory_last_seen": {
        "en": "Last time I saw your {target} was {time}, near {location}.",
        "fr": "La dernière fois que j'ai vu votre {target} c'était {time}, près de {location}.",
        "ar": "آخر مرة رأيت فيها {target} كانت {time}، بالقرب من {location}.",
        "tn": "آخر مرة شفت {target} متاعك كانت {time}، قريب من {location}.",
    },
    "memory_last_seen_no_location": {
        "en": "Last time I saw your {target} was {time}. {scene}",
        "fr": "La dernière fois que j'ai vu votre {target} c'était {time}. {scene}",
        "ar": "آخر مرة رأيت فيها {target} كانت {time}. {scene}",
        "tn": "آخر مرة شفت {target} متاعك كانت {time}. {scene}",
    },
    "memory_never_seen": {
        "en": "I have not seen your {target} yet. Ask me to find it and I will remember next time.",
        "fr": "Je n'ai pas encore vu votre {target}. Demandez-moi de le trouver et je m'en souviendrai.",
        "ar": "لم أرَ {target} بعد. اطلب مني إيجاده وسأتذكر في المرة القادمة.",
        "tn": "ما شفتش {target} متاعك بعد. قولي نلقاه وباش نتذكر المرة الجاية.",
    },

    # --- Danger / hazard --------------------------------------
    "danger_caution_ahead": {
        "en": "Caution ahead.",
        "fr": "Attention devant vous.",
        "ar": "انتبه أمامك.",
        "tn": "رد بالك قدامك.",
    },
    "danger_stop": {
        "en": "Stop now.",
        "fr": "Arrêtez-vous.",
        "ar": "توقف الآن.",
        "tn": "أوقف توا.",
    },
    "danger_move_left": {
        "en": "Move to your left.",
        "fr": "Déplacez-vous vers la gauche.",
        "ar": "تحرك إلى اليسار.",
        "tn": "زيد لليسار.",
    },
    "danger_move_right": {
        "en": "Move to your right.",
        "fr": "Déplacez-vous vers la droite.",
        "ar": "تحرك إلى اليمين.",
        "tn": "زيد لليمين.",
    },
    "danger_wait": {
        "en": "Wait.",
        "fr": "Attendez.",
        "ar": "انتظر.",
        "tn": "ستنى.",
    },
    "danger_object_position": {
        "en": "{label} {dist} on your {position}.",
        "fr": "{label} {dist} sur votre {position}.",
        "ar": "{label} {dist} على {position}ك.",
        "tn": "{label} {dist} على {position}ك.",
    },
}


# ── Lookup tables ──────────────────────────────────────────────

_TURNS: Dict[str, Dict[str, str]] = {
    "left":   {"en": "Turn slightly to your left",
               "fr": "Tournez légèrement à gauche",
               "ar": "استدر قليلاً إلى اليسار",
               "tn": "دور شويا على اليسار"},
    "center": {"en": "It is right in front of you",
               "fr": "C'est juste devant vous",
               "ar": "هو مباشرة أمامك",
               "tn": "هو قدامك مباشرة"},
    "right":  {"en": "Turn slightly to your right",
               "fr": "Tournez légèrement à droite",
               "ar": "استدر قليلاً إلى اليمين",
               "tn": "دور شويا على اليمين"},
}

_DISTANCES: Dict[str, Dict[str, str]] = {
    "close":  {"en": "close",   "fr": "près",    "ar": "قريب",  "tn": "قريب"},
    "near":   {"en": "close",   "fr": "près",    "ar": "قريب",  "tn": "قريب"},
    "nearby": {"en": "nearby",  "fr": "à proximité", "ar": "قريب", "tn": "قريب"},
    "medium": {"en": "medium distance",
               "fr": "à distance moyenne",
               "ar": "على بعد متوسط",
               "tn": "بعيد شويا"},
    "far":    {"en": "far",     "fr": "loin",    "ar": "بعيد",  "tn": "بعيد"},
}

_POSITIONS: Dict[str, Dict[str, str]] = {
    "left":   {"en": "left",   "fr": "gauche",  "ar": "اليسار", "tn": "اليسار"},
    "center": {"en": "center", "fr": "centre",  "ar": "الوسط",  "tn": "النص"},
    "right":  {"en": "right",  "fr": "droite",  "ar": "اليمين", "tn": "اليمين"},
}


# Comma joiner per language (Arabic uses ، for clarity)
_LIST_SEP = {"en": ", ", "fr": ", ", "ar": "، ", "tn": "، "}

def join_list(items: list, lang: str = "en") -> str:
    sep = _LIST_SEP.get(lang, ", ")
    return sep.join(items)


# Environment / scene_type translators (keep VLM's English categories
# but speak them in the user's language)
_ENV: Dict[str, Dict[str, str]] = {
    "indoor":  {"en": "indoors",  "fr": "à l'intérieur", "ar": "في الداخل",  "tn": "في الداخل"},
    "outdoor": {"en": "outdoors", "fr": "à l'extérieur", "ar": "في الخارج",  "tn": "برّا"},
    "unknown": {"en": "",         "fr": "",              "ar": "",           "tn": ""},
}
_SCENE_TYPE: Dict[str, Dict[str, str]] = {
    "street":    {"en": "street",    "fr": "rue",        "ar": "شارع",   "tn": "شارع"},
    "corridor":  {"en": "corridor",  "fr": "couloir",    "ar": "ممر",    "tn": "ممر"},
    "room":      {"en": "room",      "fr": "pièce",      "ar": "غرفة",   "tn": "بيت"},
    "crosswalk": {"en": "crosswalk", "fr": "passage piéton", "ar": "ممر مشاة", "tn": "بساج"},
    "stairs":    {"en": "stairs",    "fr": "escaliers",  "ar": "درج",    "tn": "درج"},
    "shop":      {"en": "shop",      "fr": "magasin",    "ar": "متجر",   "tn": "حانوت"},
    "other":     {"en": "place",     "fr": "endroit",    "ar": "مكان",   "tn": "بلاصة"},
}

def env_word(env: str, lang: str = "en") -> str:
    bundle = _ENV.get((env or "").lower(), _ENV["unknown"])
    return bundle.get(lang) or bundle.get("en") or ""

def scene_type_word(scene_type: str, lang: str = "en") -> str:
    bundle = _SCENE_TYPE.get((scene_type or "").lower(), _SCENE_TYPE["other"])
    return bundle.get(lang) or bundle.get("en") or scene_type