import torch
import torch.nn as nn
import torchvision.models as models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Cargar ResNet18 preentrenada
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adaptar la primera capa para imágenes en escala de grises (1 canal)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Congelar capas iniciales si deseas transfer learning más controlado (opcional)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Reemplazar la capa completamente conectada final
        self.base_model.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# Verificación rápida
if __name__ == '__main__':
    model = EmotionCNN(num_classes=7)
    print(model)

    # Simular una entrada (batch de imágenes de 48x48 en escala de grises)
    sample_input = torch.randn(4, 1, 48, 48)  # batch size = 4
    output = model(sample_input)
    print("Output shape:", output.shape)  # Debería ser (4, 7)
