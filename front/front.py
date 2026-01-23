import requests
import streamlit as st
import os

st.set_page_config(page_title="Toxic Detector", page_icon="🛡️", layout="wide")

st.title("Détection Toxicité d'un mesage")
st.caption("Cette interface appelle une seule API FastAPI (mêmes endpoints /translate et /detect).")

# -------- Sidebar config --------
with st.sidebar:
    st.header("API")
    
    api_base = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    st.markdown("---")
    st.header("Langue du message")
    source_lang = st.selectbox("Source", ["fr", "en", "es", "de", "it"], index=0)

    translate_to_en = st.checkbox("Traduire en anglais avant analyse", value=True)

    st.markdown("---")
    st.header("Seuil / Options")
    timeout = st.slider("Timeout (secondes)", 2, 30, 15)

    st.markdown("---")
    if st.button("Tester /health"):
        try:
            r = requests.get(api_base.rstrip("/") + "/health", timeout=timeout)
            st.code(r.text)
        except Exception as e:
            st.error(f"Health check failed: {e}")

# -------- Main UI --------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Message")
    text = st.text_area(
        "Texte à analyser",
        height=220,
        value="Bonjour, je teste mon détecteur."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        run_btn = st.button("Analyser", type="primary")
    with c2:
        clear_btn = st.button("Effacer")
    with c3:
        show_translate_btn = st.checkbox("Afficher aussi la traduction", value=True)

    if clear_btn:
        st.session_state["__clear__"] = True

if st.session_state.get("__clear__", False):
    st.session_state["__clear__"] = False
    st.rerun()

with col2:
    st.subheader("Résultat")
    result_area = st.empty()

def call_detect(msg: str):
    url = api_base.rstrip("/") + "/detect"
    payload = {
        "text": msg,
        "source_lang": source_lang,
        "translate_to_en": translate_to_en
    }
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()

def render_result(data: dict):
    label = data.get("label", "unknown")
    pct = float(data.get("toxic_percentage", 0.0))
    thr = float(data.get("threshold", 0.5))
    translated = data.get("translated_text", "")

    if label == "toxic":
        st.error(f"⚠️ TOXIC détecté — {pct:.2f}% (seuil {thr})")
    else:
        st.success(f"Non toxic — {pct:.2f}% (seuil {thr})")

    
    st.progress(min(max(pct / 100.0, 0.0), 1.0))

    if show_translate_btn:
        st.markdown("**Texte (après traduction EN si activée)**")
        st.text_area("Traduction / texte analysé", value=translated, height=160)

if run_btn:
    if not text.strip():
        st.warning("Mets un message")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                data = call_detect(text)
                with result_area.container():
                    render_result(data)
            except Exception as e:
                with result_area.container():
                    st.error(f"Erreur: {e}")

st.markdown("---")
st.markdown("### Exemples rapides")
ex1, ex2, ex3 = st.columns(3)
if ex1.button("Exemple neutre"):
    st.session_state["example_text"] = "Hello, how are you today?"
if ex2.button("Exemple limite"):
    st.session_state["example_text"] = "You're stupid and annoying."

if "example_text" in st.session_state:
    st.info(f"Copie-colle cet exemple dans la zone de texte :\n\n{st.session_state['example_text']}")
