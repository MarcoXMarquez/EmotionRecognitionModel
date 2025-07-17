import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

def get_loaders(data_root, batch_size=64, val_split=0.2):
    # Transformaciones comunes para entrenamiento y validación
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # <--- Esto es clave
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


    train_data = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=val_transform)

    # Asegurar balanceo de clases con muestreo ponderado
    class_counts = Counter([label for _, label in train_data])
    class_weights = [1.0 / class_counts[label] for _, label in train_data]
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    class_names = train_data.classes

    return train_loader, val_loader, class_names
