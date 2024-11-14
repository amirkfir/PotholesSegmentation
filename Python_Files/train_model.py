import torch
# from menuinst.utils import data_path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from DataLoaderFunc import *
from data_loader import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize import *
from generate_object_proposals import get_batch_selective_search_regions, evaluate_batch_object_proposals, prepare_proposals_images
from object_data_loader import load_and_transform_objects

from torchsummary import summary
import torch.optim as optim
from torchviz import make_dot

##general parameters
data_path = '../data/Potholes/'
image_resize = 512
batch_size = 50
IOU_th = 0.7
##load data
trainset, valset, testset, train_loader, val_loader, test_loader = load_and_transform_dataset(val_size=0.05,
                                                                                              batch_size=batch_size,
                                                                                              image_resize=image_resize,
                                                                                              data_path=data_path)

# test loader
images, (objects, num_objects) = next(iter(train_loader))

visualize_boxes(images, objects, num_objects)

batch_rects = get_batch_selective_search_regions(images)
visualize_proposals(images, batch_rects, num_proposals=50)
prepare_proposals_images()

object_trainset, object_testset, object_train_loader, object_test_loader = load_and_transform_objects(
                                                                                              batch_size=batch_size,
                                                                                              image_resize=image_resize)

# Display a batch of training images
imshow_batch(object_train_loader, image_resize, batch_size=16)

# run image classification on objects
