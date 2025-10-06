import re
import numpy as np
import pandas as pd
import pickle
import nltk
import flask
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from text_cleaning import RegexCleaner,TextCleaning

# Load trained pipeline
with open("sentiment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Create Flask app
app = flask.Flask(__name__)

@app.route("/")
def home():
    return flask.render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get text from HTML form
    text_input = flask.request.form["text"]

    # Make prediction
    pred = int(pipeline.predict([text_input])[0])

    # Return JSON
    return flask.jsonify({
        "prediction": pred,
    })

if __name__ == "__main__":
    app.run(debug=True)
