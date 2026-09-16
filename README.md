<!-- PORTFOLIO_CONFIG
system: PERSO
tech: FastAPI / BERT / Streamlit / Argos Translate / Docker
desc: API de détection de discours haineux basée sur un pipeline NLP combinant traduction automatique et modèle BERT fine-tuné. Le projet inclut une API FastAPI, une interface Streamlit et un système de classification avec score de toxicité en temps réel.
color: #ff0055
visible: true
debloy: false
-->


# Toxic Detector API (Translate → Detect → %)

Ce projet est un pipeline de modération automatique conçu pour détecter les discours haineux et le langage offensant. Il combine un service de traduction et un modèle de Deep Learning (BERT).

## Architecture

1.  **Traduction** (Argos Translate) : Normalisation du texte vers l'anglais.
2.  **Détection** (BERT fine-tuné) : Classification du texte.
3.  **Score** : Calcul de la probabilité de toxicité (%).
4.  **Interface** : Frontend interactif avec Streamlit.

> **Définition Toxicité** : Hate speech + Offensive language.
> Le modèle repose sur une classification binaire : Non-toxique (classes 0, 1) vs Toxique (classe 2).

---

## Fonctionnalités

### API (Backend)
- `POST /translate` : Traduit un texte (ex: fr → en).
- `POST /detect` : Pipeline complet (Traduction optionnelle → Détection → Label + Score).
- `GET /health` : État du service (modèle chargé, device, seuil de confiance).

### Frontend (Streamlit)
- Zone de texte pour l'analyse.
- Affichage du label de classification.
- Barre de progression pour le score de toxicité.
- Affichage de la traduction (si activée).

---

## Prérequis

- **Python** 3.10+ (recommandé 3.11)
- **Docker** & **Docker Compose** (Optionnel)

---

## Installation et Lancement

### 1. Entraînement du modèle
Avant de lancer l'API, il est nécessaire d'entraîner le modèle pour générer les fichiers de poids (sauvegardés dans `./model`).

Le dataset CSV provient de : Hate Speech and Offensive Language Dataset.
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

```bash
python train.py
```

### 2. Lancement de l'API (FastAPI)
```bash
cd app
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/mac:
# source .venv/bin/activate

python -m pip install -r requirements-api.txt

# Assurez-vous que le dossier ./model contient le modèle entraîné
uvicorn main:app --reload --host 0.0.0.0 --port 8000
