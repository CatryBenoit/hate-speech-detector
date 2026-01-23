import os
import socket
from contextlib import asynccontextmanager
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import argostranslate.package
import argostranslate.translate

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===================== CONFIG =====================
APP_PORT = int(os.getenv("APP_PORT", "8000"))
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.5"))

# Paires de langues minimum à vérifier au démarrage (tu peux ajouter)
REQUIRED_LANGUAGE_PAIRS: List[Tuple[str, str]] = [
    ("fr", "en"),
]

# Si True => tente d'installer automatiquement les packs manquants au démarrage (nécessite internet)
AUTO_INSTALL_MISSING = False


# ===================== UTILS =====================
def get_local_ip() -> str:
    """Essaye de récupérer l'IP locale LAN (utile pour afficher l'URL réseau)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def has_translation_pair(from_code: str, to_code: str) -> bool:
    """
    Vérifie robustement si la paire from->to existe dans Argos (compatible versions).
    """
    langs = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in langs if getattr(l, "code", None) == from_code), None)
    if not from_lang:
        return False

    translations = getattr(from_lang, "translations_from", None)
    if not translations:
        return False

    for t in translations:
        # Selon versions: t.to_lang ou t.to_language
        to_lang = getattr(t, "to_lang", None) or getattr(t, "to_language", None)
        to_lang_code = getattr(to_lang, "code", None)

        # fallback éventuel
        if to_lang_code is None:
            to_lang_code = getattr(t, "to_code", None) or getattr(t, "target_code", None)

        if to_lang_code == to_code:
            return True

    return False


def install_pair(from_code: str, to_code: str) -> bool:
    """
    Télécharge/installe le pack Argos. Nécessite internet.
    """
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    pkg = next((p for p in packages if p.from_code == from_code and p.to_code == to_code), None)
    if not pkg:
        return False
    path = pkg.download()
    argostranslate.package.install_from_path(path)
    return True


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Traduit via Argos. Nécessite que le pack source->target soit installé.
    """
    if source_lang.lower() == target_lang.lower():
        return text

    if not has_translation_pair(source_lang.lower(), target_lang.lower()):
        raise RuntimeError(f"Pack de traduction non installé pour {source_lang}->{target_lang}")

    return argostranslate.translate.translate(text, source_lang.lower(), target_lang.lower())


# ===================== MODEL LOAD =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

startup_status = {
    "ok": True,
    "missing_language_pairs": [],
    "model_loaded": False,
    "device": device,
    "model_dir": MODEL_DIR,
}


# ===================== FASTAPI (lifespan) =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    lan_ip = get_local_ip()
    local_url = f"http://127.0.0.1:{APP_PORT}"
    lan_url = f"http://{lan_ip}:{APP_PORT}"

    print("\n=== API Traduction + Toxic Detector démarrée ===")
    print(f"Local : {local_url}")
    print(f"Docs  : {local_url}/docs")
    print(f"LAN   : {lan_url}")
    print("================================================\n")

    # 1) Vérif packs Argos
    missing = []
    for src, tgt in REQUIRED_LANGUAGE_PAIRS:
        if not has_translation_pair(src, tgt):
            missing.append((src, tgt))

    if missing and AUTO_INSTALL_MISSING:
        print("Packs manquants détectés. Installation auto...")
        for src, tgt in missing:
            ok = install_pair(src, tgt)
            print(f" - {src}->{tgt}: {'OK' if ok else 'ECHEC'}")

        # re-check
        missing = [(s, t) for (s, t) in REQUIRED_LANGUAGE_PAIRS if not has_translation_pair(s, t)]

    if missing:
        startup_status["ok"] = False
        startup_status["missing_language_pairs"] = [f"{s}->{t}" for s, t in missing]
        print("⚠️ Packs Argos manquants :", ", ".join(startup_status["missing_language_pairs"]))
        print("➡️ /translate et /detect peuvent échouer si la paire est requise.\n")
    else:
        print("✅ Packs Argos requis OK\n")

    # 2) Charger le modèle Toxic
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        startup_status["model_loaded"] = True
        print(f"✅ Modèle chargé depuis {MODEL_DIR} sur {device}\n")
    except Exception as e:
        startup_status["ok"] = False
        startup_status["model_loaded"] = False
        print(f"❌ Impossible de charger le modèle depuis {MODEL_DIR} : {e}\n")

    yield

    # ---- SHUTDOWN ----
    print("Arrêt API 👋")


app = FastAPI(
    title="Translate + Toxic Detector API",
    version="3.0",
    lifespan=lifespan
)


# ===================== SCHEMAS =====================
class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "fr"
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    translated_text: str


class DetectRequest(BaseModel):
    text: str
    source_lang: str = "fr"
    translate_to_en: bool = True  # si False, on considère que text est déjà en anglais


class DetectResponse(BaseModel):
    original_text: str
    translated_text: str
    toxic_probability: float
    toxic_percentage: float
    label: str            # "toxic" | "non_toxic"
    threshold: float


# ===================== ENDPOINTS =====================
@app.get("/health")
def health():
    return {
        **startup_status,
        "threshold": TOXIC_THRESHOLD,
        "required_language_pairs": [f"{s}->{t}" for s, t in REQUIRED_LANGUAGE_PAIRS],
    }


@app.post("/translate", response_model=TranslateResponse)
def api_translate(payload: TranslateRequest):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le champ 'text' est vide.")

    try:
        out = translate_text(text, payload.source_lang, payload.target_lang)
        return TranslateResponse(translated_text=out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def predict_toxic_proba(text_en: str) -> float:
    if model is None or tokenizer is None:
        raise RuntimeError("Modèle non chargé. Vérifie /health et MODEL_DIR.")

    inputs = tokenizer(text_en, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        return float(probs[1].item())  # classe 1 = toxic


@app.post("/detect", response_model=DetectResponse)
def api_detect(payload: DetectRequest):
    original = (payload.text or "").strip()
    if not original:
        raise HTTPException(status_code=400, detail="Le champ 'text' est vide.")

    try:
        if payload.translate_to_en and payload.source_lang.lower() != "en":
            translated = translate_text(original, payload.source_lang, "en")
        else:
            translated = original

        translated = (translated or "").strip()
        if not translated:
            raise RuntimeError("Texte traduit vide.")

        p = predict_toxic_proba(translated)
        label = "toxic" if p >= TOXIC_THRESHOLD else "non_toxic"

        return DetectResponse(
            original_text=original,
            translated_text=translated,
            toxic_probability=p,
            toxic_percentage=round(p * 100, 2),
            label=label,
            threshold=TOXIC_THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
