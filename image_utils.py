from PIL import Image, ImageDraw, ImageFont
import torch
import road_quality_classifier
from torchvision import transforms, utils, datasets


def predict_class(image_path):
    # Loading the model
    model = road_quality_classifier.RoadQualityClassifier()
    model = torch.load("model/road_model.pt",map_location=torch.device('cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class and quality of the road
    road_qual_dict = {0: 'good', 1: 'poor', 2: 'satisfactory', 3: 'very_poor'}
    
    
    image = Image.open(image_path)
    image = image.convert('RGB')

    # Prediction logic
    transform_norm = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    img_normalized = transform_norm(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    with torch.no_grad():
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        class_name = road_qual_dict[index]
    return class_name
  
  
  
  
def draw_prediction(img, predict_class):
    x, y, w, h = 0, 0, 190, 55

    # Convert image to PIL format
    img = img.convert('RGB')

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