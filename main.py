import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Initialize Firebase Admin SDK
cred = credentials.Certificate('safetrix-117b2-firebase-adminsdk-gakry-63a5157e5f.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def fetch_data_from_firestore():
    data = db.collection('collected_data').stream()
    records = [doc.to_dict() for doc in data]
    return records

def preprocess_data(records):
    X = []
    y = []

    for record in records:
        features = [
            record.get('rainfall_intensity', 0.0),
            record.get('duration', 0.0),
            record.get('frequency', 0.0),
            record.get('runoff_coefficient', 0.0),
            record.get('catchment_area', 0.0),
            record.get('land_cover', 0.0),
            record.get('evaporation_rate', 0.0),
            record.get('temperature', 0.0),
            record.get('humidity', 0.0)
        ]
        
        flood_probability = (0.3 * features[0] + 0.2 * features[1] +
                             0.3 * features[3] + 0.1 * features[4] +
                             0.05 * (100 - features[8]) - 0.2 * features[6])
        
        flood_probability = (flood_probability - 0) / (200 - 0)
        label = int(flood_probability > 0.5)

        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Fetch and preprocess data
data_records = fetch_data_from_firestore()
X, y = preprocess_data(data_records)

# Load the existing model
model = tf.keras.models.load_model('flood_prediction_model.h5')

# Adjust model architecture to include dropout
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Recompile the model with a new optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the new data
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Loss: {loss:.2f}")

# Save the updated model
model.save('flood_prediction_model.keras')
print("Model saved as 'flood_prediction_model.keras'")
