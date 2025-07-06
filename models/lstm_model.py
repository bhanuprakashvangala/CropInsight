import torch
import torch.nn as nn


class CropLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=1):
        super(CropLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out
