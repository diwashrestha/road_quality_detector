import streamlit as st
import pandas as pd
import numpy as np
import cv2
# import library
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Road Quality Detection 🛣️🔎')

uploaded_file = st.file_uploader("Choose a image file", type=['jpg','png','jpeg'])




# Define a CNN Architecture
class RoadQualityClassifier(nn.Module):
    def __init__(self):
        super(RoadQualityClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=75, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
 
 
 
# Create new model and load states
model = RoadQualityClassifier()
model = torch.load("/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/model/road_model.pt",map_location=torch.device('cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class and quality of the road
road_qual_dict = {0: 'good', 1: 'poor', 2: 'satisfactory', 3: 'very_poor'}

def pred_class(image_path,model):
   img = Image.open(image_path)
   transform_norm = transforms.Compose([transforms.Resize((300,300)),
                                       transforms.ToTensor()])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      classes = road_qual_dict
      class_name = classes[index]
      return class_name
  
  
  
  
def image_edit(img, predict_class):
    x, y, w, h = 0, 0, 190, 55

    # Convert image to PIL format
    img = img.convert('RGB')

    # # Resize image
    # down_width, down_height = 500, 400
    # resized_down = img.resize((down_width, down_height), Image.LANCZOS)

    # # Create a new image with black background
    if predict_class == "good" or predict_class == "satisfactory":
        
        bg_color = (0, 108, 28)  # Black
        text_color = (255, 255, 255)
    else:
        bg_color = (190, 32, 28)  # Black
        text_color = (255, 255, 255)
        
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x + w, y + h), fill = bg_color)

    # Add text
    text_position = (x + int(w / 8), y + int(h / 4))
    font = ImageFont.truetype("arial.ttf",24)  # Example font, adjust as needed
    draw.text(text_position, predict_class, font = font,fill=text_color)

    # Display the edited image
    return(img)





if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    col1.header("Test Image")
    col1.image(opencv_image, channels="BGR")
    # Now do something with the image! For example, let's display it:
    road_quality_class = pred_class(uploaded_file,model)
    test_img = Image.open(uploaded_file)
    result_img = image_edit(test_img,road_quality_class)
    col2.header("Test Result")
    col2.image(result_img)


