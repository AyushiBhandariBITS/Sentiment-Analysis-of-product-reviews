import pandas as pd
import numpy as np
import re
import contractions
import nltk
import torch
import joblib
import streamlit as st
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# 1. VADER Model
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    return "Positive" if score > 0 else "Negative"

# 2. RoBERTa Model
roberta_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
def roberta_sentiment(text):
    result = roberta_pipeline(text)[0]
    return result['label']

# 3. RandomForest Model
# Load pre-trained RandomForest model and TF-IDF vectorizer
rf_model, vectorizer = joblib.load("random_forest_model.pkl")


def random_forest_sentiment(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    return rf_model.predict(text_vectorized)[0]

# 4. DistilBERT Model
# Load pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_model")

def distilbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = distilbert_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# Streamlit Frontend
st.title("Sentiment Analysis with 4 Models")
user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    preprocessed_text = preprocess_text(user_input)
    st.subheader("Results:")
    st.write("**VADER Sentiment:**", vader_sentiment(user_input))
    st.write("**RoBERTa Sentiment:**", roberta_sentiment(user_input))
    st.write("**Random Forest Sentiment:**", random_forest_sentiment(preprocessed_text))
    st.write("**DistilBERT Sentiment:**", distilbert_sentiment(user_input))
