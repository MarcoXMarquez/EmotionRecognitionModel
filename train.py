import torch
import torch.nn as nn
import torch.optim as optim
from models.emotion_cnn import EmotionCNN
from utils.data_loader import get_loaders

def train_model(data_dir, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_loaders(data_dir, batch_size=batch_size)

    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validación al final de cada época
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"🎯 Época {epoch+1} | Pérdida: {total_loss:.4f} | Precisión Validación: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "emotion_model.pth")
            print("💾 Modelo mejorado guardado\n")

if __name__ == "__main__":
    train_model("data")
