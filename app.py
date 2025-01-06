from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the pre-trained SVM model
model = load("svm_model_hog.joblib")

UPLOAD_FOLDER = 'static/image'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_resize_and_compute_hog_image(image_path):
    # Read and resize the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read the image from the provided path.")

    resized_img = cv2.resize(img, (128, 256))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features, _ = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        channel_axis=None
    )

    return features


def encode_image_to_base64(image_path):
    img = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Extract HOG features and predict
            features = load_resize_and_compute_hog_image(file_path)
            prediction = model.predict([features])
            label = "Paederus" if prediction[0] == 1 else "Other Ant"

            # Encode the uploaded image to Base64 for displaying
            encoded_image = encode_image_to_base64(file_path)

            return render_template(
                'result.html',
                prediction=label,
                image_data=encoded_image
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)