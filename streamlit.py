import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE  = (256, 256)
IMG_SHAPE = IMG_SIZE +(3,)

#1. load model
loaded_model = tf.keras.models.load_model('pretrained_models/EfficientNetB3_Model.h5')

#2. load class names
with open ('data/class_names', 'rb') as fp:
    class_names = pickle.load(fp)

st.header("Dog Identification Application")
# st.text_input("Enter your Name: ", key="name")

uploaded_file = st.file_uploader("Upload an image of your dog and identify its breed...", type=['jpg', 'jpeg', 'png'])

# If an image is uploaded, display it and make a prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img_g = np.expand_dims(img, axis=0)
    custom_predict = loaded_model.predict(img_g)

    # Display the uploaded image and the predictions
    # st.write(f"{class_names[np.argmax(custom_predict[0])]}  ({round(np.max(custom_predict[0]*100))}% confidence)")
    if round(np.max(custom_predict[0]*100))>50:
        st.write(f'<p style="font-size:26px; color:green;"> {class_names[np.argmax(custom_predict[0])]}  ({round(np.max(custom_predict[0]*100))}% confidence)</p> ', unsafe_allow_html=True)
    else:
        argsorts = np.argsort(custom_predict[0])
        sorts = np.sort(custom_predict[0])
        st.write(f'<p style="font-size:26px; color:red;"> {class_names[argsorts[-1]]}  ({round(sorts[-1]*100)}% confidence) OR {class_names[argsorts[-2]]}  ({round(sorts[-2]*100)}% confidence)</p> ', unsafe_allow_html=True)

    st.image(img/255,  use_column_width=True)
    st.write('Notes:')
    st.write('1. The model is a first simplest baseline version, the prediction result will be improved in later versions')
    st.write('2. The model can identify up to 120 types of dog breeds')
    option = st.selectbox(
    'All Breeds',
    class_names)

