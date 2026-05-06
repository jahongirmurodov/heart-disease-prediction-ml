from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

COLS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')

        age_years = float(data['age'])
        age_days = round(age_years * 365.25)

        gender = float(data['gender'])
        height = float(data['height'])
        weight = float(data['weight'])
        ap_hi = float(data['ap_hi'])
        ap_lo = float(data['ap_lo'])
        cholesterol = float(data['cholesterol'])
        gluc = float(data['gluc'])
        smoke = float(data['smoke'])
        alco = float(data['alco'])
        active = float(data['active'])

        features = pd.DataFrame(
            [[age_days, gender, height, weight, ap_hi, ap_lo,
              cholesterol, gluc, smoke, alco, active]],
            columns=COLS
        )

        features_scaled = scaler.transform(features)
        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability_healthy': round(float(probability[0]) * 100, 1),
            'probability_disease': round(float(probability[1]) * 100, 1),
            'patient_name': f"{first_name} {last_name}".strip(),
            'age_days': age_days
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
