import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load("nb_model.pkl")              # <-- Updated
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # <-- Updated

# Hardcoded stopwords (no need for nltk.download)
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

lemmatizer = WordNetLemmatizer()

# Clean input text
def clean_text(text):
    text = re.sub(r"\W", " ", str(text)).lower()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)

# Streamlit App
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste a news article below, and the model will predict whether it's **Fake** or **Real**.")

user_input = st.text_area("Enter news article here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned])  # <-- Add this step
        prediction = model.predict(vectorized_input)[0]
        label = "🟢 Real News" if prediction == 0 else "🔴 Fake News"  # Your model: 0 = real, 1 = fake
        
        st.markdown(f"## Prediction: {label}")
