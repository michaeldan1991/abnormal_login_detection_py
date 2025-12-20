from flask import Flask, request, jsonify
from dotenv import load_dotenv
import joblib
import os
import pandas as pd
from constants import MODELS

app = Flask(__name__)

# Function to extract time-based features
def extract_time_features(x):
    x['hour'] = x['Timestamp'].dt.hour
    x['day_of_week'] = x['Timestamp'].dt.dayofweek
    return x[['hour', 'day_of_week']]

# Load model from environment variable
load_dotenv(dotenv_path="config.env")
model_type = os.getenv("MODEL_TYPE", "")
if model_type not in MODELS:
    raise ValueError(f"Model invalid: {model_type}")
info = MODELS[model_type]
print(f"Loading {info['name']} from {info['path']}")
model = joblib.load(info["path"])
threshold = info["threshold"]

@app.route('/fraud/predict', methods=['POST'])
def predict():
    try:
    # Request data to predict
        data = request.get_json()

        # Mapping data to DataFrame
        columns = ['User ID', 'Timestamp', 'Login Status', 'IP Address', 'Device Type',
            'Location', 'Session Duration', 'Failed Attempts', 'Behavioral Score']
        df = pd.DataFrame([data['features']], columns=columns)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # Predict
        probs = model.predict_proba(df)[0][1]
        prediction = int(probs >= threshold)

        # Response
        return jsonify({
            "code": 200,
            "message": "Success",
            "data": {
                "prediction": prediction,
                "probability": float(probs),
                "model": info['value']
            }
        }), 200
    except Exception as e:
        print(f"Has error: {e}")
        return jsonify({
            "code": 400,
            "message": str(e)
        }), 400

@app.route('/fraud/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok"
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)