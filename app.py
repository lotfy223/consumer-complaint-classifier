from pathlib import Path

import joblib
import streamlit as st
import hashlib
import io
import requests

st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon="📨",
    layout="wide",
)


@st.cache_resource
def load_model():
    url = (
        "https://drive.google.com/uc?export=download"
        "&id=1GGGIS_7Zle6LmxYKm2GpVa1qrt_Be5_V"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    model_bytes = response.content

    expected_hash = (
        "25da4ae0e5193de5c5d06694de6d93c"
        "700a5d96bf185cdd4ef0ee6ace64896b2"
    )

    actual_hash = hashlib.sha256(model_bytes).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            "The downloaded model does not match the expected file."
        )

    return joblib.load(io.BytesIO(model_bytes))


model = load_model()

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(110deg, #4f46e5, #9333ea, #db2777);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { color: white; }
    </style>

    <div class="hero">
        <h1>📨 Consumer Complaint Classifier</h1>
        <p>Classify a consumer complaint into one of five categories</p>
        <p>TF-IDF + Logistic Regression · Test Macro F1: 0.8635</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.subheader("Complaint narrative")
    text = st.text_area(
        "Enter a complaint in English",
        height=220,
        placeholder="Describe the issue here...",
    )
    submitted = st.button("Classify", type="primary")

with right:
    st.subheader("Prediction")

    if submitted and not text.strip():
        st.warning("Please enter a complaint first.")

    elif submitted:
        probabilities = model.predict_proba([text])[0]
        scores = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        best_label, best_score = scores[0]
        st.success(f"Predicted category: {best_label.title()}")
        st.metric("Model score", f"{best_score * 100:.1f}%")

        st.write("**Scores for all categories**")
        for label, score in scores:
            st.write(f"{label.title()} — {score * 100:.1f}%")
            st.progress(float(score))
    else:
        st.info("Enter a complaint and click Classify.")

st.caption(
    "Prototype: the model always chooses one of five categories, "
    "even for unrelated text. Scores are not guaranteed to be "
    "calibrated. Please do not enter sensitive personal information."
)
