import streamlit as st
import pandas as pd
import numpy as np
import cv2
import road_classify as rc

st.title('Road Quality Detection')

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    print(rc.classify_image(opencv_image,rc.model))


