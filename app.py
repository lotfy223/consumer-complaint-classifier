from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon="📨",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "baseline.joblib")


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
