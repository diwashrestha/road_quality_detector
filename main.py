import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from road_quality_classifier import RoadQualityClassifier
from image_utils import predict_class, draw_prediction
import random



st.set_page_config(layout="wide")
st.title('Road Quality Detection 🛣️🔍📝')

uploaded_file = st.file_uploader("Choose a image file", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    try:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
        col1, col2 = st.columns(2)
        col1.header("Test Image")
        col1.image(opencv_image, channels="BGR")
        # Now do something with the image! For example, let's display it:
        test_img = Image.open(uploaded_file)
        road_quality_class = predict_class(uploaded_file)
        result_img = draw_prediction(test_img,road_quality_class)
        col2.header("Test Result")
        col2.image(result_img)
    except Exception as e:
        st.error(f"An error occurred: {e}",icon="🚨")

else:
    st.subheader("Try Sample Images")
    
    image_deck = [
        "test_image/autobahn.jpg",
        "test_image/bad_road.jpg",
        "test_image/sample5.jpg",
        "test_image/sample1.jpg"
    ]
    
    
    if st.button('Try'):
        image_number = random.randint(0,3) 
        # Use a sample image
        sample_image_path = image_deck[image_number]  # Provide the path to your sample image
        sample_image = Image.open(sample_image_path)

        col1, col2 = st.columns(2)
        col1.header("Sample Image")
        col1.image(sample_image, channels="RGB")

        # Perform prediction on the sample image
        road_quality_class = predict_class(sample_image_path)
        result_img = draw_prediction(sample_image, road_quality_class)
        col2.header("Sample Result")
        col2.image(result_img)
        
    
    col1, col2, col3, col4 = st.columns(4)
    col1.image(image_deck[0],channels="BGR")
    col2.image(image_deck[1],channels="BGR")
    col3.image(image_deck[2],channels="BGR")
    col4.image(image_deck[3],channels="BGR")

