from flask import Flask, jsonify, request
from decimal import Decimal
import joblib
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate, Dropout, Flatten
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ALLOWED_EXTENSIONS = set(['jpg','png','heic','jpeg'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def preprocess_input_image_prediction(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Add an extra dimension for batch
    return img_array

# Function to make predictions
def predict_image(model, img_path):
    input_image = preprocess_input_image_prediction(img_path)
    predictions = model.predict([input_image, input_image])
    return predictions




@app.route("/diabetes", methods=["POST"])
def predict_diabetes():
    if request.method == "POST":
        print(request.files)
        # Load the pre-trained model
        loaded_model = load_model('best_model1.h5')

        # Get the image file from the request
        image_file = request.files['image']

        # Save the image to a temporary file
        temp_image_path = 'temp_image.jpg'
        image_file.save(temp_image_path)

        # Example usage to predict an image
        predictions = predict_image(loaded_model, temp_image_path)

        class_labels = ['diabetes', 'non-diabetes']  # Modify these labels according to your classes
        predicted_class = class_labels[np.argmax(predictions)]
        print(f'Predicted Class: {predicted_class}')
        print(f'Prediction Probabilities: {predictions}')

        # Return the prediction in JSON format
        return jsonify({"code": 200, "has_diabetes": predicted_class})

    return jsonify({"error": "Request error"}, 400)


@app.route("/test", methods=["GET"])
def predict():
    if request.method == "GET":
        loaded_model = load_model('best_model1.h5')
        image_path_to_predict = '1.jpg'
        predictions = predict_image(loaded_model, image_path_to_predict)

        class_labels = ['diabetes', 'non-diabetes']  # Modify these labels according to your classes
        predicted_class = class_labels[np.argmax(predictions)]
        print(f'Predicted Class: {predicted_class}')
        print(f'Prediction Probabilities: {predictions}')
        return jsonify({"code": 200, "has_diabetes":predicted_class})

    return jsonify({"error": "Request error"}, 400)


if __name__ == "__main__":
    app.run(debug=True)
