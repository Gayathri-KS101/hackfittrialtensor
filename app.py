from flask import Flask, request, render_template, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import tensorflow as tf
import subprocess

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('safetrix-117b2-firebase-adminsdk-gakry-63a5157e5f.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Load the TensorFlow model
model = tf.keras.models.load_model('flood_prediction_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-data', methods=['POST'])
def submit_data():
    # Retrieve all parameters from the form
    data = {
        'rainfall_intensity': float(request.form.get('rainfall_intensity')),
        'duration': float(request.form.get('duration')),
        'frequency': float(request.form.get('frequency')),
        'runoff_coefficient': float(request.form.get('runoff_coefficient')),
        'catchment_area': float(request.form.get('catchment_area')),
        'land_cover': float(request.form.get('land_cover')),
        'evaporation_rate': float(request.form.get('evaporation_rate')),
        'temperature': float(request.form.get('temperature')),
        'humidity': float(request.form.get('humidity'))
    }

    # Add the received data to Firestore
    db.collection('collected_data').add(data)
    return "Data submitted successfully"

@app.route('/predict', methods=['GET'])
def predict():
    # Call main.py to fetch and preprocess data, and train the model
    subprocess.call(['python', 'main.py'])

    # Fetch and preprocess data
    records = fetch_data_from_firestore()
    X, _ = preprocess_data(records)

    if X.size == 0:
        return jsonify({"error": "No valid data to predict"}), 400

    # Make predictions
    predictions = model.predict(X)
    probabilities = predictions.flatten()

    # Prepare the result
    results = [{'probability': float(prob), 'prediction': 'Flood likely' if prob > 0.5 else 'Flood unlikely'} for prob in probabilities]

    return jsonify(results)

def fetch_data_from_firestore():
    data = db.collection('collected_data').stream()
    records = [doc.to_dict() for doc in data]
    return records

def preprocess_data(records):
    X = []
    for record in records:
        features = [
            record.get('rainfall_intensity', 0),
            record.get('duration', 0),
            record.get('frequency', 0),
            record.get('runoff_coefficient', 0),
            record.get('catchment_area', 0),
            record.get('land_cover', 0),
            record.get('evaporation_rate', 0),
            record.get('temperature', 0),
            record.get('humidity', 0)
        ]
        X.append(features)
    return np.array(X), None

if __name__ == '__main__':
    app.run(debug=True)
