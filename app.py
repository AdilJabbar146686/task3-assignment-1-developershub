import streamlit as st
import nltk
import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (only runs once per deployment)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app layout
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below and click **Predict**.")

user_input = st.text_area("Enter text here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vectorized_input = vectorizer.transform([processed])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ The news is **REAL**.")
        else:
            st.error("🚨 The news is **FAKE**.")
