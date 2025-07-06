import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.cnn_model import CropCNN
from utils.data_loader import CropImageDataset
from utils.digits_dataset import DigitsDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Try using the original dataset if it exists, otherwise fall back to the
# built-in sklearn digits dataset which keeps this example self contained.
dataset_root_folder = os.path.join(
    "data/images",
    "Sign-Language-Digits-Dataset-master",
)
dataset_root = os.path.join(dataset_root_folder, "Dataset")
csv_path = "data/labels.csv"
use_digits_dataset = False

if os.path.exists(dataset_root_folder):
    if not os.path.exists(csv_path) and os.path.exists(dataset_root):
        print("Generating labels.csv from existing dataset ...")
        with open(csv_path, "w") as f:
            f.write("image,label\n")
            for label in sorted(os.listdir(dataset_root)):
                label_dir = os.path.join(dataset_root, label)
                if not os.path.isdir(label_dir):
                    continue
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        f.write(f"{label}/{img_name},{label}\n")
else:
    use_digits_dataset = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if use_digits_dataset:
    print("Using sklearn digits dataset ...")
    dataset = DigitsDataset(transform=transform)
else:
    dataset = CropImageDataset(csv_file="data/labels.csv", root_dir=dataset_root, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropCNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total if total else 0
    print(
        f"Epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f}, val_acc={val_acc:.4f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/crop_cnn.pth")

print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

