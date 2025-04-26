import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
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
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Load model and vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app layout
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter a news article or headline:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vectorized_input = vectorizer.transform([processed])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 1:
            st.success("✅ This news is likely **REAL**.")
        else:
            st.error("⚠️ This news is likely **FAKE**.")
