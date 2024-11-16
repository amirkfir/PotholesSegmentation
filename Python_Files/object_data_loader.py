import os
import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


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
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        x = self.transform(image) if self.transform else image

        return x, y, image_path


def load_and_transform_objects(batch_size, image_resize):
    data_path = '../data/Potholes/Proposals/'

    target_size = (image_resize, image_resize)
    random_state = 42

    train_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    classes_order = ["Potholes", "NotPotholes"]

    train_set = CustomImageFolder(
        root=os.path.join(data_path, 'train'),
        classes_order=classes_order,
        transform=train_transforms
    )

    test_set = CustomImageFolder(
        root=os.path.join(data_path, 'test'),
        classes_order=classes_order,
        transform=test_transforms
    )

    test_set_extended = PathExtendedDataset(
        train=False, 
        transform=test_transforms, 
        data_path=data_path)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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
