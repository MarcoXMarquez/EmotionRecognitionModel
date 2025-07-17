
import torch
from sklearn.metrics import classification_report
from models.emotion_cnn import EmotionCNN
from utils.data_loader import get_loaders

def evaluate_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, classes = get_loaders(data_dir)
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("emotion_model.pth"))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    evaluate_model("data")
