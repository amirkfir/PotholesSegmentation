import torch
# from menuinst.utils import data_path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from DataLoaderFunc import *
from data_loader import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.flatten()

for i in range(4):
    ax = axs[i]
    ax.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

    for j in range(num_objects[i]):
        xmin = objects[i][j][0]
        ymin = objects[i][j][1]
        xmax = objects[i][j][2]
        ymax = objects[i][j][3]
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=8, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)

plt.tight_layout()
plt.show()

##load/create model


# run train loop
