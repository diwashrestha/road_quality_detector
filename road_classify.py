import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import torch.utils.data as data
from sklearn.metrics import classification_report, confusion_matrix


from PIL import Image
from PIL import ImageDraw

model = torch.load('/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/model/road_model.pt')
model.eval()




def classify_image(image_path,model):
   img = image_path
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
      classes = idx2class
      class_name = classes[index]
      return class_name



def image_class(image_path, predict_class):
    img = cv2.imread(image_path)
    x,y,w,h = 0,350,190,55

    # Draw black background rectangle
    #cv2.rectangle(img, (x, x), (x + w, y + h), (0,0,0), -1)

    
    down_width = 500
    down_height = 400
    down_points = (down_width, down_height)
    resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
    
    # Add text
    cv2.putText(resized_down, predict_class, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return(plt.imshow(resized_down))