
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes