import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def calculate_mean_std(dataset):
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in loader:
        data = data[0]  # data[0] because data is a tuple of (images, labels)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

# Define your transformations (exclude normalization and data augmentation)
transforms_for_calculation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
data_path = '../data/Potholes/Proposals/'
# Create a dataset with these transformations
dataset_for_calculation = datasets.ImageFolder(
    root=os.path.join(data_path, 'train'),
    transform=transforms_for_calculation
)

# Calculate mean and std
mean, std = calculate_mean_std(dataset_for_calculation)

print(f"Calculated mean: {mean}")
print(f"Calculated std: {std}")