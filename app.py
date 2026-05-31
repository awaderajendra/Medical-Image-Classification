import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model_path = 'brain_cancer_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None

classes = ['Cancerous', 'Non-Cancerous']

@app.route('/', methods=['GET'])
def index():
    return "Brain Cancer Detection API is Running. Use /predict endpoint to upload image."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        os.remove(file_path)

        return jsonify({
            'prediction': classes[class_idx],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)