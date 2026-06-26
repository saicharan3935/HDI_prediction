from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Page
@app.route("/Prediction")
def prediction():
    return render_template("indexnew.html")


# Predict Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input values
        input_features = [float(x) for x in request.form.values()]

        # Convert to numpy array
        features = np.array(input_features).reshape(1, -1)

        # Predict
        prediction = model.predict(features)
        pred = int(prediction[0])

        # Convert prediction into text
        if pred == 0:
            result = "LOW HUMAN DEVELOPMENT"
        elif pred == 1:
            result = "MEDIUM HUMAN DEVELOPMENT"
        elif pred == 2:
            result = "HIGH HUMAN DEVELOPMENT"
        elif pred == 3:
            result = "VERY HIGH HUMAN DEVELOPMENT"
        elif pred == 4:
            result = "VERY HIGH HUMAN DEVELOPMENT"
        else:
            result = "UNKNOWN"

        return render_template(
            "resultnew.html",
            prediction_text=f"Prediction : {result}"
        )

    except Exception as e:
        return render_template(
            "resultnew.html",
            prediction_text=f"Error : {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)