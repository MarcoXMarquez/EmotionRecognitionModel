import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from models.model import EmotionCNN
from utils.data_loader import get_loaders
from tqdm import tqdm

def train_model(data_dir, epochs=500, batch_size=64, patience=7):  # 🔁 500 épocas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Dispositivo detectado: {device}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ La carpeta '{data_dir}' no existe. Asegúrate de que la ruta es correcta.")

    # Carga de datasets
    dataset_names = os.listdir(data_dir)
    print(f"📁 Cargando datos desde '{data_dir}' con {len(dataset_names)} datasets...")
    for dataset_name in tqdm(dataset_names, desc="Cargando datasets", unit="dataset"):
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ Dataset {dataset_name} no encontrado en {dataset_path}")

    train_loader, val_loader, classes = get_loaders(data_dir, batch_size=batch_size)
    print(f"✅ Datos cargados correctamente. Clases detectadas: {classes}")

    # Verificar distribución de clases
    def check_class_distribution(loader):
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.numpy())
        return Counter(all_labels)

    train_counter = check_class_distribution(train_loader)
    print("\n📊 Distribución de clases en entrenamiento:")
    for emotion, count in zip(classes, train_counter.values()):
        print(f"  - {emotion}: {count} muestras")

    # Pesos para clases desbalanceadas
    class_counts = torch.tensor(list(train_counter.values()), dtype=torch.float)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()

    print("\n⚙️ Configurando modelo y optimizador...")
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    no_improve = 0

    print("\n🚀 Entrenamiento iniciado...\n")

    # Barra de progreso para las épocas
    with tqdm(range(epochs), desc="Épocas", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0

            loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} (train)", leave=False)
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item())

            train_acc = correct_train / total_train

            # Validación
            model.eval()
            correct_val, total_val, val_loss = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_acc = correct_val / total_val
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(val_acc)

            epoch_bar.set_postfix(
                train_acc=f"{train_acc*100:.2f}%",
                val_acc=f"{val_acc*100:.2f}%",
                train_loss=f"{total_loss/len(train_loader):.4f}",
                val_loss=f"{avg_val_loss:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), "emotion_model.pth")
                print(f"💾 Mejor modelo guardado con precisión de validación: {val_acc*100:.2f}%")
            else:
                no_improve += 1
                print(f"📉 Sin mejora. Intentos sin mejora: {no_improve}/{patience}")
                if no_improve >= patience:
                    print("🚨 Early stopping activado. Fin del entrenamiento.")
                    break

    print(f"\n🏁 Entrenamiento finalizado. Mejor precisión en validación: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model("data_combined")
