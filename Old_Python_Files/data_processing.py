import os
import glob
import random
from PIL import Image
import numpy as np
import torch
# from torchvision import transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class PH2(torch.utils.data.Dataset):
    def __init__(self, shared_transform, image_only_transform, image_paths, label_paths):
        'Initialization'
        # Transforms applied to both image and label
        self.shared_transform = shared_transform
        # Transforms applied only to image
        self.image_only_transform = image_only_transform
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.shared_transform is not None:
            image, label = self.shared_transform(image, label)

        if self.image_only_transform is not None:
            image = self.image_only_transform(image)

        label = T.ToTensor()(label)

        return image, label


def load_and_transform_dataset(val_size, test_size, batch_size, image_resize, data_path='/content/drive/MyDrive/datasets/hotdog_nothotdog'):

    rand_rot_angle = 10
    rand_crop_size = 350
    flip_probability = 0.7
    gaussian_blur_kernel_size = 5
    color_jitter_brightness = 0.2
    color_jitter_contrast = 0.2
    color_jitter_saturation = 0.2
    color_jitter_hue = 0.05
    target_size = (image_resize, image_resize)
    
    random_state = 42

    image_paths, label_paths = load_data_paths(data_path)

    train_img_paths, test_img_paths, train_label_paths, test_label_paths = train_test_split(
        image_paths, label_paths, test_size=test_size, random_state=random_state)

    train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(
        train_img_paths, train_label_paths, test_size=val_size, random_state=random_state)

    avg_mean, std_avg = calculate_normalization_params(train_img_paths)

    print(f"avg_mean: {avg_mean}")
    print(f"std_avg: {std_avg}")

    # shared transforms are applied both to labels and images
    train_shared_transform = T.Compose([
        T.RandomHorizontalFlip(p=flip_probability),
        T.RandomVerticalFlip(p=flip_probability),
        T.RandomRotation(rand_rot_angle),
        T.Resize(target_size),
    ])

    test_shared_transform = T.Compose([
        T.Resize(target_size),
    ])

    # Image-only transforms
    train_image_only_transform = T.Compose([
        T.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue
        ),
        T.GaussianBlur(gaussian_blur_kernel_size),
        T.ToTensor(),
        GaussianNoise(mean=0.0, std=0.1),
        T.Normalize(mean=avg_mean, std=std_avg),
    ])

    test_image_only_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=avg_mean, std=std_avg),
    ])
    
    trainset_unprocessed_transform = T.Compose([
        T.ToTensor(),
    ])

    trainset = PH2(
        shared_transform=train_shared_transform,
        image_only_transform=train_image_only_transform,
        image_paths=train_img_paths,
        label_paths=train_label_paths
    )
    
    trainset_unprocessed = PH2(
        shared_transform=trainset_unprocessed_transform,
        image_only_transform=trainset_unprocessed_transform,
        image_paths=train_img_paths,
        label_paths=train_label_paths
    )

    valset = PH2(
        shared_transform=test_shared_transform,
        image_only_transform=test_image_only_transform,
        image_paths=val_img_paths,
        label_paths=val_label_paths
    )

    testset = PH2(
        shared_transform=test_shared_transform,
        image_only_transform=test_image_only_transform,
        image_paths=test_img_paths,
        label_paths=test_label_paths
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return trainset, valset, testset, train_loader, val_loader, test_loader#, trainset_unprocessed


# def load_data_paths(data_path):
#     image_paths = glob.glob(data_path + '/images/*')
#     label_paths = glob.glob(data_path + '/masks/*')
#
#     image_paths.sort(key=lambda x: os.path.basename(x).split('_')[0])
#     label_paths.sort(key=lambda x: os.path.basename(x).split('.')[0])
#     return image_paths, label_paths
#
# def calculate_normalization_params(train_img_paths):
#     all_pixels = []
#
#     for train_image_path in train_img_paths:
#         train_image = Image.open(train_image_path)
#         tensor_image = T.ToTensor()(train_image)
#         pixels = tensor_image.view(tensor_image.size(0), -1)
#         all_pixels.append(pixels)
#
#     all_pixels = torch.cat(all_pixels, dim=1)
#
#     mean = torch.mean(all_pixels, dim=1)
#     std = torch.std(all_pixels, dim=1)
#
#     return mean, std
