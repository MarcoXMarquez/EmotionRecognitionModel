
import torch
from sklearn.metrics import classification_report
from models.model import EmotionCNN
from utils.data_loader import get_loaders

def evaluate_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_loaders(data_dir)
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    evaluate_model("data_combined")
