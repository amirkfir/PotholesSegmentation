import os
import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import torch

import cv2
import numpy as np
from PIL import ImageOps, Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


class HistogramEqualization(object):
    def __call__(self, img):
        return ImageOps.equalize(img)

class CLAHE(object):
    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl,a,b))
        img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_eq)

class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, classes_order, transform=None):
        super(CustomImageFolder, self).__init__(root, transform=transform)

        original_class_to_idx = self.class_to_idx.copy()

        self.classes = classes_order
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(self.classes)}

        mapping = {original_class_to_idx[cls_name]                   : self.class_to_idx[cls_name] for cls_name in self.classes}

        new_samples = []
        for path, original_cls_idx in self.samples:
            new_cls_idx = mapping[original_cls_idx]
            new_samples.append((path, new_cls_idx))

        self.samples = new_samples
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            # Albumentations expects image in NumPy format
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert to tensor if no transform is provided
            image = torch.tensor(image).permute(2, 0, 1)

        return image, label


class PathExtendedDataset(Dataset):
    def __init__(self, train, transform=None, data_path='./data/Proposals'):
        'Initialization'
        self.transform = transform

        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(
            data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: 1 - id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        # image = Image.open(img_path)
        c = os.path.split(os.path.split(img_path)[0])[1]
        label = self.name_to_label[c]

        if self.transform:
            # Albumentations expects image in NumPy format
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert to tensor if no transform is provided
            image = torch.tensor(image).permute(2, 0, 1)

        return image, label, img_path


def load_and_transform_objects(batch_size, image_resize, data_path = '../data/Potholes/Proposals/', only_test=False):
    # data_path = '../data/Potholes/Proposals/'

    target_size = (image_resize, image_resize)
    gaussian_blur_kernel_size = 5
    color_jitter_brightness = 0.3
    color_jitter_contrast = 0.3
    color_jitter_saturation = 0.1
    color_jitter_hue = 0.15
    random_state = 42

    #ImageNet values
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]

    #Our train set values
    mean = [0.5090, 0.4893, 0.4666]
    std = [0.1386, 0.1351, 0.1310]

    train_transforms = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5
        ),
        #A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        #A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    classes_order = ["Potholes", "NotPotholes"]

    if not only_test:
        train_set = CustomImageFolder(
            root=os.path.join(data_path, 'train'),
            classes_order=classes_order,
            transform=train_transforms
        )
    else:
        train_set = None

    test_set = CustomImageFolder(
        root=os.path.join(data_path, 'test'),
        classes_order=classes_order,
        transform=test_transforms
    )

    test_set_extended = PathExtendedDataset(
        train=False, 
        transform=test_transforms, 
        data_path=data_path)

    if not only_test:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = None

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    test_loader_extended = DataLoader(
        test_set_extended,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_set, test_set, test_set_extended, train_loader, test_loader, test_loader_extended