import torch
# from menuinst.utils import data_path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from DataLoaderFunc import *
from data_loader import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize import visualize_boxes, visualize_proposals
from generate_object_proposals import get_batch_selective_search_regions

from torchsummary import summary
import torch.optim as optim
from torchviz import make_dot

##general parameters
data_path = '../data/Potholes/'
image_resize = 512
batch_size = 32

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

##load/create model


# run train loop
