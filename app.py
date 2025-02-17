from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf  # Import TensorFlow
from tensorflow import keras  # Import Keras from TensorFlow
import os

app = Flask(__name__)

# Load the trained model (use absolute path if necessary for deployment)
try:
    model = keras.models.load_model("brain_tumor_detector.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle the case where the model fails to load

# Class names (Make SURE these match your training data directory names EXACTLY!)
class_names = ['no_tumor', 'glioma', 'meningioma', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load. Check the console for errors.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Check if the image was loaded successfully
        if img is None:
            return jsonify({'error': 'Error decoding image.  Make sure it is a valid image file.'}), 400

        # Resize and preprocess the image (use the same size as your training data!)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)  # Create a batch
        img = img / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img)
        class_index = np.argmax(predictions[0])
        predicted_class = class_names[class_index]
        confidence = float(predictions[0][class_index])  # Convert to float for JSON

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Prediction error: {e}")  # Log the error for debugging
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# Update the port to use Render's dynamic port (10000 in most cases)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))  # Default to 5000 for local development



