import torch
import torch.nn as nn
import torchvision.models as models


class CropCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CropCNN, self).__init__()
        self.base_model = models.resnet18(weights=None)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
