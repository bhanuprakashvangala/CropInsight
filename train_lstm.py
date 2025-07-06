import torch
from torch.utils.data import DataLoader
from models.lstm_model import CropLSTM
from utils.data_loader import SequenceDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Example random data (replace with real sequence data later)
sequences = np.random.rand(100, 12, 10)
labels = np.random.rand(100, 1)

dataset = SequenceDataset(sequences, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CropLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    model.train()
    loop = tqdm(loader)
    for seqs, lbls in loop:
        seqs, lbls = seqs.to(device), lbls.to(device)
        outputs = model(seqs)
        loss = criterion(outputs, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/2]")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'models/crop_lstm.pth')
