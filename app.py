# app.py for Streamlit deployment

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and vectorizer using absolute paths
model_path = os.path.join(script_dir, 'spam_model.pkl')
vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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