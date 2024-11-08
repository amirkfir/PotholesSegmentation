import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
from PIL import Image
import torch
from fontTools.afmLib import readlines
from sympy.physics.units import length
from sympy.printing.pretty.pretty_symbology import annotated
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import xml.etree.ElementTree as ET
from torchvision.transforms import v2 as T


def prepare_labels(xml_path, out_path, max_objects):
    for filename in os.listdir(xml_path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(xml_path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        objects = []
        for child in root:
            if child.tag == 'object':
                object = np.array([int(child[4][0].text),
                                   int(child[4][1].text),
                                   int(child[4][2].text),
                                   int(child[4][3].text)])
                objects.append(object)
            elif child.tag == 'size':
                pic_size = (int(child[0].text), int(child[1].text))
        out = np.zeros((max_objects, 4))
        out[0:len(objects)] = objects

        out_directory = os.path.dirname(out_path + "/" + filename[:-4] + ".pkl")
        os.makedirs(out_directory, exist_ok=True)

        with open(out_path + "/" + filename[:-4] + ".pkl", 'wb') as fp:
            pickle.dump((torch.Tensor(out), len(objects), pic_size), fp)
    print(max_objects)


class Potholes(torch.utils.data.Dataset):
    def __init__(self, target_only_transform, image_only_transform, image_paths, label_paths):
        'Initialization'
        # Transforms applied to both image and label
        self.target_only_transform = target_only_transform
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
        with open(label_path, 'rb') as fp:
            label = pickle.load(fp)

            if self.target_only_transform is not None:
                label = self.target_only_transform(label)

            if self.image_only_transform is not None:
                image = self.image_only_transform(image)

            #  label = T.Compose([
            #      T.ToImage(),
            #      T.ToDtype(torch.float32, scale=True)
            #  ])(label)

            return image, label


# def collect(batch):
#     images = []
#     targets = []
#     for image, target in batch:
#         images.append(image)
#         targets.append(target)
#     images = torch.stack(images, dim=0)  # Stack images along the batch dimension
#     return images, targets


def load_data_paths(data_path):
    with open(data_path + 'splits.json', 'r') as file:
        files = json.load(file)

    train_files = files["train"]
    test_files = files["test"]

    train_image_paths = [f"{data_path}annotated-images/{x[:-4]}.jpg" for x in train_files]
    train_objects_paths = [f"{data_path}objects_files/{x[:-4]}.pkl" for x in train_files]
    test_image_paths = [f"{data_path}annotated-images/{x[:-4]}.jpg" for x in test_files]
    test_objects_paths = [f"{data_path}objects_files/{x[:-4]}.pkl" for x in test_files]

    train_image_paths.sort(key=lambda x: os.path.basename(x).split('_')[0])
    train_objects_paths.sort(key=lambda x: os.path.basename(x).split('.')[0])
    test_image_paths.sort(key=lambda x: os.path.basename(x).split('_')[0])
    test_objects_paths.sort(key=lambda x: os.path.basename(x).split('.')[0])
    return train_image_paths, train_objects_paths, test_image_paths, test_objects_paths


class BBox_Resize(torch.nn.Module):
    def __init__(self, dest_shape):
        super().__init__()
        self.dest_shape = dest_shape

    def forward(self, target):
        boxes, t, org_shape = target
        # Compute the scaling ratios
        ratio = torch.tensor([
            self.dest_shape[0] / org_shape[0],
            self.dest_shape[1] / org_shape[1],
            self.dest_shape[0] / org_shape[0],
            self.dest_shape[1] / org_shape[1]
        ])
        # Apply the scaling ratios to the bounding boxes
        boxes = torch.round(boxes * ratio.unsqueeze(0))
        return (boxes, t)


def load_and_transform_dataset(val_size, batch_size, image_resize,
                               data_path='../data/Potholes/'):
    # rand_rot_angle = 10
    # rand_crop_size = 350
    # flip_probability = 0.7
    # gaussian_blur_kernel_size = 5
    # color_jitter_brightness = 0.2
    # color_jitter_contrast = 0.2
    # color_jitter_saturation = 0.2
    # color_jitter_hue = 0.05
    target_size = (image_resize, image_resize)

    random_state = 42

    image_mask_pairs = []

    # for file in train_files:
    #     image_path = os.path.join(data_path, "annotated-images", f"{file[-4]}.jpg")
    #     objects_path = os.path.join(data_root, "objects_files", f"{file[-4]}.pkl")
    #     image_mask_pairs.append((image_path, objects_path))

    trainable_image_paths, trainable_objects_paths, test_image_paths, test_objects_paths = load_data_paths(data_path)

    train_image_paths, val_image_paths, train_objects_paths, val_objects_paths = train_test_split(
        trainable_image_paths, trainable_objects_paths, test_size=val_size, random_state=random_state)

    # avg_mean, std_avg = calculate_normalization_params(train_img_paths)

    # print(f"avg_mean: {avg_mean}")
    # print(f"std_avg: {std_avg}")

    # shared transforms are applied both to labels and images
    # train_shared_transform = T.Compose([
    #     T.RandomHorizontalFlip(p=flip_probability),
    #     T.RandomVerticalFlip(p=flip_probability),
    #     T.RandomRotation(rand_rot_angle),
    #     T.Resize(target_size),
    # ])
    #
    train_targets_transform = T.Compose([
        BBox_Resize(target_size), ])
    test_targets_transform = T.Compose([
        BBox_Resize(target_size), ])
    #
    # # Image-only transforms
    # In your load_and_transform_dataset function
    train_image_only_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(target_size),
    ])

    test_image_only_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(target_size),
    ])

    train_shared_transform = None
    # train_image_only_transform = None
    test_shared_transform = None
    # test_image_only_transform = None

    trainset = Potholes(
        target_only_transform=train_targets_transform,
        image_only_transform=train_image_only_transform,
        image_paths=train_image_paths,
        label_paths=train_objects_paths
    )

    valset = Potholes(
        target_only_transform=test_targets_transform,
        image_only_transform=test_image_only_transform,
        image_paths=val_image_paths,
        label_paths=val_objects_paths
    )

    testset = Potholes(
        target_only_transform=test_targets_transform,
        image_only_transform=test_image_only_transform,
        image_paths=test_image_paths,
        label_paths=test_objects_paths
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=3, collate_fn=None)
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return trainset, valset, testset, train_loader, val_loader, test_loader


data_root = '../data/Potholes/'

# prepare label dataset
prepare_labels(data_root + "annotated-images", data_root + "objects_files", 19)

# with open("/home/amir/Documents/Autonomous Systems Master/DLICV/object_detection/poster1/data/Potholes/Potholes/objects_files/img-399.pkl",'rb') as file:
#     a = pickle.load(file)

##read file splitting:


## prepare dataloader
