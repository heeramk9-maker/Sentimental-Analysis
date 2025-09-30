import joblib

def clean_text(text: str) -> str:
    return text.lower()

def load_models():
    bnb = joblib.load("models/bernoulli_nb_model.pkl")
    svm = joblib.load("models/svm_model.pkl")
    logreg = joblib.load("models/logreg_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return bnb, svm, logreg, vectorizer
