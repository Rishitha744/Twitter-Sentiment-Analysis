import streamlit as st
import joblib
import torch
import re
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# ── Text cleaning (same as notebook) ─────────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource
def load_tfidf():
    pipeline = joblib.load("models/best_tfidf_model.joblib")
    return pipeline

@st.cache_resource
def load_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_model")
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/distilbert_model"
    )
    model.eval()
    return tokenizer, model

# ── Prediction functions ──────────────────────────────────────────────────────
def predict_tfidf(text, pipeline):
    cleaned = clean_tweet(text)
    pred = pipeline.predict([cleaned])[0]
    prob = pipeline.predict_proba([cleaned])[0]
    return pred, round(float(max(prob)) * 100, 1)

def predict_distilbert(text, tokenizer, model):
    cleaned = clean_tweet(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = round(float(probs[pred]) * 100, 1)
    return pred, confidence

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐦 Tweet Sentiment Analyzer")
st.markdown(
    "This app classifies the sentiment of a tweet as **Positive** or **Negative** "
    "using two models trained in our CSCE 676 Data Mining project — a classical "
    "TF-IDF + Logistic Regression model and a fine-tuned DistilBERT transformer."
)

st.divider()

# Tweet input
tweet = st.text_area(
    "Enter a tweet below:",
    placeholder="e.g. I just had the best coffee of my life!",
    height=100
)

analyze = st.button("Analyze Sentiment", type="primary")

if analyze:
    if not tweet.strip():
        st.warning("Please enter a tweet first.")
    else:
        # Load models
        with st.spinner("Loading models..."):
            tfidf_pipeline         = load_tfidf()
            distilbert_tok, distilbert_model = load_distilbert()

        # Predictions
        tfidf_pred,  tfidf_conf  = predict_tfidf(tweet, tfidf_pipeline)
        bert_pred,   bert_conf   = predict_distilbert(
            tweet, distilbert_tok, distilbert_model
        )

        label_map = {0: "Negative 😞", 1: "Positive 😊"}
        color_map = {0: "red", 1: "green"}

        st.divider()
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### TF-IDF + Logistic Regression")
            st.markdown(
                f"**Sentiment:** :{color_map[tfidf_pred]}[{label_map[tfidf_pred]}]"
            )
            st.markdown(f"**Confidence:** {tfidf_conf}%")
            st.progress(tfidf_conf / 100)

        with col2:
            st.markdown("#### DistilBERT (fine-tuned)")
            st.markdown(
                f"**Sentiment:** :{color_map[bert_pred]}[{label_map[bert_pred]}]"
            )
            st.markdown(f"**Confidence:** {bert_conf}%")
            st.progress(bert_conf / 100)

        st.divider()

        # Cleaned tweet preview
        with st.expander("See cleaned tweet text"):
            st.code(clean_tweet(tweet))

st.divider()
