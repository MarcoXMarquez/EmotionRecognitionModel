import torch
import torch.nn as nn
import torch.optim as optim
from models.emotion_cnn import EmotionCNN
from utils.data_loader import get_loaders

def train_model(data_dir, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🚀 Iniciando entrenamiento en dispositivo: {device}")
    if device.type == "cuda":
        print(f"🖥️ GPU detectada: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"🔋 Memoria total de GPU: {mem:.0f} MB\n")
    else:
        print("⚠️ Usando CPU. El entrenamiento será más lento.\n")

    train_loader, classes = get_loaders(data_dir, batch_size=batch_size)
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        print(f"🧪 Época {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  ▶️ Lote {batch_idx+1}/{len(train_loader)} - Pérdida: {loss.item():.4f}")

        print(f"✅ Fin de época {epoch+1} - Pérdida total: {total_loss:.4f}\n")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("💾 Modelo guardado como emotion_model.pth\n")

if __name__ == "__main__":
    train_model("data")
