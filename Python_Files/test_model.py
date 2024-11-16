import torch
import torchvision.models as models

from data_loader import *
from object_data_loader import load_and_transform_objects
from model import Pothole_RCNN


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_resize = 128
batch_size = 50

object_trainset, object_testset, object_train_loader, object_test_loader = load_and_transform_objects(
                                                                                                  batch_size=batch_size,
                                                                                                  image_resize=image_resize)



# create model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
resnet18 = models.resnet18(pretrained=True)
model = Pothole_RCNN(num_classes, resnet18).to(device)

model.load_state_dict(torch.load('rcnn_model.pth'))







torch.cuda.empty_cache()
