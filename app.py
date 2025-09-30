import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_text, load_models

# =====================
# Load models & vectorizer
# =====================
bnb, svm, logreg, vectorizer = load_models()

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📝 Sentiment Analysis App")
st.write("Compare Naive Bayes, SVM, and Logistic Regression on the Sentiment140 dataset")

# =====================
# User input
# =====================
user_input = st.text_area("Enter a sentence:", "I love this product!")

if st.button("Predict"):
    clean = clean_text(user_input)
    tfidf = vectorizer.transform([clean])

    # Predictions + confidence
    preds = {
        "Naive Bayes": (bnb.predict(tfidf)[0], bnb.predict_proba(tfidf)[0][1]),
        "SVM": (svm.predict(tfidf)[0], svm.decision_function(tfidf)[0]),
        "Logistic Regression": (logreg.predict(tfidf)[0], logreg.predict_proba(tfidf)[0][1])
    }

    st.subheader("🔮 Predictions with Confidence")
    for model, (pred, conf) in preds.items():
        sentiment = "Positive 😀" if pred == 1 else "Negative 😡"
        if model == "SVM":
            st.write(f"**{model}** → {sentiment} (Score: {conf:.4f})")
        else:
            st.write(f"**{model}** → {sentiment} (Confidence: {conf:.2f})")

    # Bar chart of confidence
    st.subheader("📊 Confidence Scores")
    models = list(preds.keys())
    confs = [conf for _, conf in preds.values()]
    fig, ax = plt.subplots()
    ax.bar(models, confs, color=["skyblue", "orange", "green"])
    ax.set_ylabel("Confidence / Score")
    st.pyplot(fig)
