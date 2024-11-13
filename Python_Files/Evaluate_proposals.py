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
from generate_object_proposals import get_batch_selective_search_regions, evaluate_batch_object_proposals

from torchsummary import summary
import torch.optim as optim
from torchviz import make_dot

##general parameters
data_path = '../data/Potholes/'
image_resize = 512
batch_size = 50
num_of_batches = 10
IOU_th = 0.7
max_proposals_list = [150,300,450,600,750,900,1050,1200,1350,1500,1650,1800,1950,2100,2250,2400]

##load data
trainset, valset, testset, train_loader, val_loader, test_loader = load_and_transform_dataset(val_size=0.05,
                                                                                              batch_size=batch_size,
                                                                                              image_resize=image_resize,
                                                                                              data_path=data_path)
MABO_list = np.zeros((batch_size*num_of_batches,len(max_proposals_list)))
RecallRate_list = np.zeros((batch_size*num_of_batches,len(max_proposals_list)))

for batch in range(num_of_batches):

    # test loader
    images, (objects, num_objects) = next(iter(train_loader))
    batch_rects = get_batch_selective_search_regions(images)


    for i,max_proposals in enumerate(max_proposals_list):
        MABO, RecallRate = evaluate_batch_object_proposals((objects, num_objects),batch_rects,IOU_th,max_proposals=max_proposals)
        MABO_list[batch*batch_size:(batch+1)*batch_size,i] = MABO
        RecallRate_list[batch*batch_size:(batch+1)*batch_size,i] = RecallRate


plt.figure(figsize = (20,10))
plt.suptitle('Evaluating different number of proposals, Selective Search, num_of_images: 500, IOU TH = 0.7 ',fontsize=30)
plt.subplot(1, 2, 1)
plt.plot(max_proposals_list,MABO_list.mean(axis=0))
plt.ylabel("MABO Score",fontsize=20)
plt.xlabel("number of proposals",fontsize=20)
plt.subplot(1, 2, 2)
plt.plot(max_proposals_list,RecallRate_list.mean(axis = 0))
plt.ylabel("Recall Rate",fontsize=20)
plt.xlabel("number of proposals",fontsize=20)
plt.savefig("comparsion.jpg")
##load/create model


# run train loop
