from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        # Extract features
        fisik = float(data['fisik'])
        teknik_langkah_kaki = float(data['teknik_langkah_kaki'])
        forehand = float(data['forehand'])
        backhand = float(data['backhand'])
        teknik_permainan = float(data['teknik_permainan'])
        semangat_berlatih = float(data['semangat_berlatih'])

        # Standardize features
        features = np.array([[fisik, teknik_langkah_kaki, forehand, backhand, teknik_permainan, semangat_berlatih]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Convert prediction to string (if necessary)
        prediction_str = str(prediction[0])

        return jsonify({'prediction': prediction_str})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
