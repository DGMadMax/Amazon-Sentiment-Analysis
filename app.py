from flask import Flask, request, render_template, jsonify, send_file
import pickle
import os
import re
import numpy as np
import pandas as pd
import nltk
import base64
import matplotlib
matplotlib.use("Agg")  # Use Agg backend to prevent GUI issues
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from io import BytesIO

# Ensure necessary NLTK data is downloaded
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Define model file paths
MODEL_PATH = "Models/model_xgb.pkl"
SCALER_PATH = "Models/scaler.pkl"
VECTORIZER_PATH = "Models/countVectorizer.pkl"

# Check if model files exist
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, VECTORIZER_PATH]):
    raise FileNotFoundError("One or more required model files are missing!")

# Load model, scaler, and vectorizer
print("Loading model and preprocessing tools...")
predictor = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
cv = pickle.load(open(VECTORIZER_PATH, "rb"))
print("Model loaded successfully!")

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('i.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(data)
            response = send_file(predictions, mimetype="text/csv", as_attachment=True, download_name="Predictions.csv")
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response

        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def preprocess_text(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)


def single_prediction(text_input):
    processed_text = preprocess_text(text_input)
    X_vectorized = cv.transform([processed_text]).toarray()
    X_scaled = scaler.transform(X_vectorized)
    prediction = predictor.predict(X_scaled)
    return "Positive" if prediction[0] == 1 else "Negative"


def bulk_prediction(data):
    data["Processed_Text"] = data["Sentence"].apply(preprocess_text)
    X_vectorized = cv.transform(data["Processed_Text"]).toarray()
    X_scaled = scaler.transform(X_vectorized)
    data["Predicted Sentiment"] = predictor.predict(X_scaled)
    data["Predicted Sentiment"] = data["Predicted Sentiment"].map({1: "Positive", 0: "Negative"})
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    graph = generate_graph(data)
    return predictions_csv, graph


def generate_graph(data):
    fig = plt.figure(figsize=(5,5))
    colors = ["green", "red"]
    data["Predicted Sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=colors, startangle=90)
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph

if __name__ == "__main__":
    app.run(debug=True)
