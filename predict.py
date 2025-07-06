import argparse
import os
import torch
from models.cnn_model import CropCNN
from models.lstm_model import CropLSTM
from torchvision import transforms
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description="Run inference on a single image")
parser.add_argument("--image", type=str, help="Path to image for classification")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn_model = CropCNN(num_classes=10).to(device)
lstm_model = CropLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1).to(device)

try:
    cnn_model.load_state_dict(torch.load('models/crop_cnn.pth', map_location=device))
    lstm_model.load_state_dict(torch.load('models/crop_lstm.pth', map_location=device))
except Exception as e:
    print(f"Failed to load weights: {e}")

cnn_model.eval()
lstm_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from utils.digits_dataset import DigitsDataset

if args.image and os.path.exists(args.image):
    img = Image.open(args.image).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
else:
    sample_dataset = DigitsDataset(transform=transform)
    img, _ = sample_dataset[0]
    img = img.unsqueeze(0).to(device)

with torch.no_grad():
    output = cnn_model(img)
    pred = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred.item()}")

sequence = np.random.rand(1, 12, 10)
sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)

with torch.no_grad():
    forecast = lstm_model(sequence_tensor)
    print(f"Predicted yield: {forecast.item()}")

