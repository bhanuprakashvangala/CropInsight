import torch
from torch.utils.data import DataLoader, random_split
from models.lstm_model import CropLSTM
from utils.data_loader import SequenceDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Example random data (replace with real sequence data later)
sequences = np.random.rand(200, 12, 10)
labels = np.random.rand(200, 1)

dataset = SequenceDataset(sequences, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CropLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader)
    for seqs, lbls in loop:
        seqs, lbls = seqs.to(device), lbls.to(device)
        outputs = model(seqs)
        loss = criterion(outputs, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seqs, lbls in val_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            outputs = model(seqs)
            val_loss += criterion(outputs, lbls).item()
    val_loss /= len(val_loader)
    print(
        f"Epoch {epoch+1}: train_loss={running_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'models/crop_lstm.pth')

print(f"Training complete. Best validation loss: {best_loss:.4f}")

