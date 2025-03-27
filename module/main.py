import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def load_and_predict(model_path, img_path, target_size=(224, 224)):
    # Load the model_yolo
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model.predict(img_array)

    return predictions

if __name__ == '__main__':
    # Example usage
    model_path = '../model/image_classifier_v8.h5'
    img_path = 'cam_vuot.jpeg'
    predictions = load_and_predict(model_path, img_path)
    res= np.argmax(predictions)
    print(res)