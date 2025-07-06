import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn_model import CropCNN
from utils.data_loader import CropImageDataset
from utils.download import download_and_extract
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Remote dataset URL
image_zip_url = "https://codeload.github.com/ardamavi/Sign-Language-Digits-Dataset/zip/refs/heads/master"

# Download dataset and labels
dataset_root_folder = os.path.join(
    "data/images",
    "Sign-Language-Digits-Dataset-master",
)
if not os.path.exists(dataset_root_folder):
    download_and_extract(image_zip_url, "data/images/", force=True)

# Generate labels.csv if not present
dataset_root = os.path.join(dataset_root_folder, "Dataset")
csv_path = "data/labels.csv"
if not os.path.exists(csv_path) and os.path.exists(dataset_root):
    print("Generating labels.csv from dataset ...")
    with open(csv_path, "w") as f:
        f.write("image,label\n")
        for label in sorted(os.listdir(dataset_root)):
            label_dir = os.path.join(dataset_root, label)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    f.write(f"{label}/{img_name},{label}\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CropImageDataset(csv_file='data/labels.csv', root_dir=dataset_root, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CropCNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    model.train()
    loop = tqdm(loader)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/2]")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'models/crop_cnn.pth')
