from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model khi khởi động server
model = joblib.load("model/login_pipeline.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận JSON từ client (Spring Boot gửi qua)
    data = request.get_json()
    print(f"Received data: {data}")


    # Tạo DataFrame với đúng tên cột
    columns = ['User ID', 'Timestamp', 'Login Status', 'IP Address', 'Device Type',
           'Location', 'Session Duration', 'Failed Attempts', 'Behavioral Score']

    df = pd.DataFrame([data['features']], columns=columns)
    # Dự đoán
    df = pd.DataFrame([data['features']], columns=columns)

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0].tolist()

    # Trả kết quả
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": probability
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)