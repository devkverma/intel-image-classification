import pickle
import cv2 as cv
import requests
import streamlit as st
import numpy as np
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
from PIL import Image

model = pickle.load(open('model.pkl','rb'))

def predict(image):
    image = cv.resize(image,(100,100))
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)

    image = image / 255.0

    image = np.expand_dims(image,axis=0)

    y_pred = model.predict(image)

    class_names = ['Buildings','Forest','Glacier','Mountain','Sea','Street']

    max_confidence = np.argmax(y_pred)

    return class_names[max_confidence],y_pred[0][max_confidence]


st.title("Intel Image Classification")
st.write("Upload an image to get the prediction of the lanscape (Buildings, Forest, Glacier, Mountain, Sea, Street) in the image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image_url = st.text_input("Or enter the image URL:")

_,mid_col,_ = st.columns(3)

image_array = None

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    mid_col.image(image, caption='Uploaded Image', width = 256)

    image_array = np.array(image)
    
    image_array = cv.cvtColor(image_array,cv.COLOR_RGB2BGR)

elif image_url:

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        mid_col.image(image, caption='Image from URL', width = 256)
        image_array = np.array(image)
        image_array = cv.cvtColor(image_array,cv.COLOR_RGB2BGR)
    except Exception as e:
        st.write(f"Error loading image from the given URL")
        img_array = None

else:
    image_array = None

_,left_col,mid_col,_,_ = st.columns(5)

if image_array is not None:

    prediction = predict(image_array)
    left_col.write("Prediction:")
    mid_col.write(prediction[0])
    left_col.write("Confidence %")
    mid_col.write(prediction[1]*100)