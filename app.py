from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# ==========================
# Load Saved Files
# ==========================

with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# ==========================
# Home Route
# ==========================

@app.route("/")
def home():
    return "😊 Customer Review Sentiment Analysis API is Live!"


# ==========================
# Prediction Route
# ==========================

@app.route("/predict", methods=["POST"])
def predict():
    try:

        # Get JSON request
        data = request.get_json()

        # Check input
        if "review" not in data:
            return jsonify({
                "status": "error",
                "message": "Please provide 'review' in JSON."
            }), 400

        review = data["review"]

        # Convert review to TF-IDF features
        review_vector = tfidf.transform([review])

        # Predict sentiment
        prediction = model.predict(review_vector)

        # Convert encoded label to text
        sentiment = label_encoder.inverse_transform(prediction)

        return jsonify({
            "status": "success",
            "review": review,
            "predicted_sentiment": sentiment[0]
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ==========================
# Run Application
# ==========================

if __name__ == "__main__":
    app.run(debug=True)