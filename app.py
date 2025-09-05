# app.py for Streamlit deployment

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing function (same as in notebook)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title('Email Spam Detector')

# Text area for user input
email_text = st.text_area('Enter the email content here:')

# Predict button
if st.button('Predict'):
    if email_text:
        # Preprocess the input text
        processed_text = preprocess_text(email_text)
        # Transform the input text
        text_vectorized = vectorizer.transform([processed_text])
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        # Display result
        if prediction == 1:
            st.error('This email is SPAM.')
        else:
            st.success('This email is NOT SPAM.')
    else:
        st.warning('Please enter some email content to predict.')