# src/utils_data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
def get_cifar10(batch=128, workers=2):

    data_root = "data/cifar-10" if os.path.exists("data/cifar-10") else "/kaggle/working"
    train_tf = transforms.Compose([
        transforms.Resize(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([transforms.Resize(48), transforms.ToTensor()])

    train_set = datasets.CIFAR10(data_root, train=True, download=True , transform=train_tf)
    test_set  = datasets.CIFAR10(data_root, train=False, download=True , transform=test_tf)

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Train samples:", len(train_set), "Test samples:", len(test_set))
    return train_loader, test_loader

