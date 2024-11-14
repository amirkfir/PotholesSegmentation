import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, classes_order, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

        self.classes = classes_order
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        new_samples = []
        for path, original_cls_idx in self.samples:
            original_cls_name = self.classes[original_cls_idx]
            new_cls_idx = self.class_to_idx[original_cls_name]
            new_samples.append((path, new_cls_idx))

        self.samples = new_samples
        self.targets = [s[1] for s in self.samples]

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

    classes_order = ["Pothols", "NotPothols"]

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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory = True
    )

    # Create the testing DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    return train_set, test_set, train_loader, test_loader