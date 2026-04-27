import torch
from torchvision import datasets, transforms

class MNISTDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)