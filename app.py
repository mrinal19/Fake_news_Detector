from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import string
import os

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None


# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- LOAD MODEL ----------------
def load_model():
    global model, vectorizer

    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if len(text) < 10:
            return jsonify({"error": "Text too short"})

        processed = preprocess_text(text)
        vec = vectorizer.transform([processed])

        prob = model.predict_proba(vec)[0]

        real_prob = prob[0] * 100
        fake_prob = prob[1] * 100

        if fake_prob > 60:
            label = "FAKE"
        elif real_prob > 60:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        confidence = max(real_prob, fake_prob)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "fake_probability": round(fake_prob, 2),
            "real_probability": round(real_prob, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- START ----------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)