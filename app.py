from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load("logistic_regression_segment_model.pkl")
scaler = joblib.load("scaler_for_models.pkl")


# Define demographic features as inputs
FEATURES = ['SEC', 'FEH', 'MT', 'SEX', 'AGE', 'EDU', 'HS', 'CHILD', 'CS', 'Affluence Index']

# Mapping cluster index to segment name
cluster_names = {
    0: "Heavy Buyers / Variety Seekers",
    1: "Switchers / Promo Shoppers",
    2: "Loyalists"
}


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_segment():
    try:
        # Determine input source: JSON or form
        if request.content_type == 'application/json':
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()

        input_df = pd.DataFrame([input_data])
        for col in FEATURES:
            input_df[col] = input_df[col].astype(float)

        # Preprocess
        scaled_input = scaler.transform(input_df[FEATURES])

        # Predict
        cluster_idx = model.predict(scaled_input)[0]
        segment = cluster_names.get(cluster_idx, "Unknown Segment")

        # Return result
        if request.content_type == 'application/json':
            return jsonify({"segment": segment})
        else:
            return f"<h2>Predicted Segment: {segment}</h2>"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
