# app/inf_video.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from functools import lru_cache

# Constants
img_height, img_width = 180, 180
class_labels = ['asthmatic', 'healthy']
threshold = 0.5  # Confidence threshold

def resize_and_pad(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]), Image.Resampling.LANCZOS)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill='black')

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != (img_height, img_width):
            img = resize_and_pad(img, (img_height, img_width))
        img_array = np.array(img)
    return img_array

@lru_cache(maxsize=4)
def load_tf_model(model_path: str):
    # lru_cache will cache by model_path string
    return tf.keras.models.load_model(model_path)

def predict_single_image(image_path, model_path):
    model = load_tf_model(model_path)
    img_array = preprocess_image(image_path) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    predicted_class_index = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class_index])
    return predicted_class_index, confidence

def majority_class(predictions):
    class_counts = [0] * len(class_labels)
    for pred_class, confidence in predictions:
        if confidence >= threshold:
            class_counts[pred_class] += 1
    majority_index = int(np.argmax(class_counts))
    return class_labels[majority_index]

def predict_frames_in_directory(directory, model_path):
    frame_predictions = []
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(directory, filename)
            pred_class, confidence = predict_single_image(image_path, model_path)
            frame_predictions.append((pred_class, confidence))
    majority = majority_class(frame_predictions) if frame_predictions else "unknown"
    return majority

def predict_from_two_directories(directory1, directory2, model_path):
    frame_predictions = []
    for directory in (directory1, directory2):
        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(directory, filename)
                pred_class, confidence = predict_single_image(image_path, model_path)
                frame_predictions.append((pred_class, confidence))
    majority = majority_class(frame_predictions) if frame_predictions else "unknown"
    return majority
