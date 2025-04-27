import streamlit as st
import nltk
import string
import joblib
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---- Ensure necessary NLTK data is available ----
nltk_packages = ['punkt', 'stopwords']

for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}') if package == 'punkt' else nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package)

# ---- Load your trained model and vectorizer ----
model = joblib.load('nb_model.pkl')           # <-- Corrected filename here
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # <-- Corrected filename here

# ---- Preprocessing function ----
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# ---- Streamlit UI ----
st.title('Fake News Detection App')

input_news = st.text_area('Enter the news text')

if st.button('Predict'):
    if input_news.strip() == '':
        st.warning('Please enter some text.')
    else:
        # Preprocess
        transformed_news = transform_text(input_news)
        
        # Vectorize
        vector_input = tfidf_vectorizer.transform([transformed_news])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Display
        if result == 0:
            st.success('Prediction: **Real News** 📰✅')
        else:
            st.error('Prediction: **Fake News** 🚫📰')

