# import library
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt



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
  


def image_edit(image_path, predict_class):
    img = cv2.imread(image_path)
    x,y,w,h = 0,0,190,55

    # Draw black background rectangle
    #cv2.rectangle(img, (x, x), (x + w, y + h), (0,0,0), -1)

    
    down_width = 500
    down_height = 400
    down_points = (down_width, down_height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
    
    # Draw black background rectangle
    bg_color = (0, 0, 0)  # Black
    cv2.rectangle(resized_down, (x, y), (x + w, y + h), bg_color, -1)
    
    # Add text
    cv2.putText(resized_down, predict_class, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return(plt.imshow(resized_down))  


# test_image_paths = ["/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/autobahn.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/autobahn2.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/delhi-mumbai-expressway.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/highlands.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/indian_road.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/bad_road.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/bad_road2.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/potholes.jpg",
#                     "/home/diwashrestha/projects/GahiroPadhahi/road_quality_detector/test_image/contryside.jpg"]



# for image_path in test_image_paths:
#     predict_class = pre_image(image_path,model)
#     print(image_path + ":" +  predict_class)
#     image_class(image_path,predict_class)