import torch
import torchvision.models as models

from data_loader import *
from object_data_loader import load_and_transform_objects
from model import Pothole_RCNN

from classification_manager import get_classification_results

from generate_object_proposals import generate_and_save_proposals




image_resize = 128
batch_size = 1

object_trainset, object_testset, object_testset_extended, object_train_loader, object_test_loader, object_test_loader_extended = load_and_transform_objects(
                                                                                                  batch_size=batch_size,
                                                                                                  image_resize=image_resize)

generate_and_save_proposals(out_path_bboxes='../data/Potholes/Proposals_test', subsets_to_prepare=['test'])


# for images, labels in object_test_loader:
#     # print(images)
#     print("\n\n")
#     print(labels)

#     print("\n\n")
#     break

# for test_obj, label in object_test_loader_extended:
#     img_path = test_obj[0]
#     img = test_obj[1]
#     print(img_path)
#     print(img)
#     print(label)

# for dataset_obj, dataloader_obj in zip(object_testset_extended, object_test_loader_extended):
#     print(dataset_obj)
#     print(dataloader_obj)

#     print("\n\n")

