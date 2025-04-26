import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---- Download NLTK data ----
nltk.download('punkt')
nltk.download('stopwords')

# ---- Load artifacts ----
model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ---- NLTK setup ----
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---- Preprocessing ----
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(stemmer.stem(w)) for w in tokens if w not in stop_words]
    return " ".join(cleaned)

# ---- Streamlit UI ----
st.title("📰 Fake News Detector")
st.write("Enter a news article below and click **Predict**:")

user_input = st.text_area("News Article", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        processed = preprocess(user_input)
        vect = vectorizer.transform([processed])
        pred = model.predict(vect)[0]
        label = "REAL" if pred == 1 else "FAKE"
        st.markdown(f"## Prediction: **{label}**")
