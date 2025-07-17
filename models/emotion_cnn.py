import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 48 -> 46
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 46 -> 44
        self.pool = nn.MaxPool2d(2, 2)  # reduce a la mitad: 44 -> 22
        self.dropout = nn.Dropout(0.25)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))  # fuerza salida fija 4x4

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
