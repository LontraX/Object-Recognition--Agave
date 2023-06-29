from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
app = Flask(__name__)

# Load the serialized model
model = tf.keras.models.load_model('detection_model_01.h5')

# Define the API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400
        
        # Get the image file from the request
        image_file = request.files['image']
        
        # Read the image file using OpenCV
        image_array = np.array(cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), -1))

        # Resize the image to the required shape
        resized_image = cv2.resize(image_array, (224, 224))

        # Make predictions using the loaded model
        predictions = model.predict(np.expand_dims(resized_image, axis=0))

        # Postprocess the predictions if necessary
        postprocessed_predictions = postprocess_predictions(predictions)
        
        # Return the predictions as a response
        return jsonify({'predictions': postprocessed_predictions})
    
    except Exception as e:
        # Handle any exceptions that occur during prediction
        return jsonify({'error': str(e)}), 400
    
# Postprocess the predictions
def postprocess_predictions(predictions):
    # Convert the predictions into class labels
    predicted_labels = np.argmax(predictions, axis=1)
    # For example, convert predictions to a suitable format, apply thresholding, etc.
    postprocessed_predictions = predicted_labels.tolist()
    
    return postprocessed_predictions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
