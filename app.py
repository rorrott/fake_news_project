from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
from xgboost import XGBClassifier
import joblib



app = Flask(__name__)

# Load the trained model
with open('models/xgb_model.pkl', 'rb') as f:
    xgb_model_model = pickle.load(f)

with open('tidf/tfidf_headline.pkl', 'rb') as f:
    tfidf_headline = pickle.load(f)

with open('tidf/tfidf_article.pkl', 'rb') as f:
    tfidf_article = pickle.load(f)


def     clean_text(text):

    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    return text

def     tokenize_and_stem(text):
    
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
    return " ".join(tokens)



app = Flask(__name__)

# Mapping of numeric labels to class names
label_mapping = {0: 'unrelated', 1: 'agree', 2: 'discuss', 3: 'disagree'}

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if not hasattr(app.config, 'xgb_model_model'):
        app.config['xgb_model_model'] = joblib.load('models/xgb_model.pkl')

    new_headline = request.form['headline']
    new_article_body = request.form['articleBody']

    # Preprocess new data
    new_headline_cleaned = clean_text(new_headline)
    new_article_body_cleaned = clean_text(new_article_body)
    new_headline_preprocessed = tokenize_and_stem(new_headline_cleaned)
    new_article_body_preprocessed = tokenize_and_stem(new_article_body_cleaned)

    # Transform new data using TF-IDF
    new_headline_tfidf = tfidf_headline.transform([new_headline_preprocessed])
    new_article_body_tfidf = tfidf_article.transform([new_article_body_preprocessed])

    # Combine features
    new_data_combined = hstack((new_headline_tfidf, new_article_body_tfidf))

    # Make prediction
    new_prediction = app.config['xgb_model_model'].predict(new_data_combined)
    prediction = label_mapping[new_prediction[0]]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
