import torch
from models.cnn_model import CropCNN
from models.lstm_model import CropLSTM
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.download import download_file


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn_model = CropCNN(num_classes=2).to(device)
lstm_model = CropLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1).to(device)

cnn_weights_url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
lstm_weights_url = "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/expected_flow.pt"

download_file(cnn_weights_url, "models/crop_cnn.pth")
download_file(lstm_weights_url, "models/crop_lstm.pth")

try:
    cnn_model.load_state_dict(torch.load('models/crop_cnn.pth', map_location=device, weights_only=False))
    lstm_model.load_state_dict(torch.load('models/crop_lstm.pth', map_location=device, weights_only=False))
except Exception as e:
    print(f"Failed to load weights: {e}")

cnn_model.eval()
lstm_model.eval()

image_path = "data/images/Sign-Language-Digits-Dataset-master/Dataset/0/IMG_1118.JPG"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = cnn_model(img)
    pred = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred.item()}")

sequence = np.random.rand(1, 12, 10)
sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)

with torch.no_grad():
    forecast = lstm_model(sequence_tensor)
    print(f"Predicted yield: {forecast.item()}")
