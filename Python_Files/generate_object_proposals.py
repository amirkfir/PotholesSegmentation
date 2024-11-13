import cv2
import numpy as np
import os
import pickle
import torch

from data_loader import load_and_transform_dataset
from pandas.core.computation.expr import intersection

def get_selective_search_regions(image):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    return rects


def get_batch_selective_search_regions(images):
    batch_rects = []
    for image in images:
        rects = get_selective_search_regions(image)
        batch_rects.append(rects)
    return batch_rects

def IOU(box1,box2):
    max_x1 = np.maximum(box1[0], box2[0])
    max_y1 = np.maximum(box1[1], box2[1])
    min_x2 = np.minimum(box1[2], box2[2])
    min_y2 = np.minimum(box1[3], box2[3])
    if max_x1 > min_x2 or max_y1 > min_y2:
        return 0
    else:
        intersection = (min_y2-max_y1) * (min_x2-max_x1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (box1_area + box2_area - intersection)

def evaluate_batch_object_proposals(groundtruth_batch, proposals_batch,IOU_th,max_proposals):
    MABO = []
    RecallRate = []
    for img_idx, proposals in enumerate(proposals_batch):
        groundtruth = groundtruth_batch[0][img_idx,:groundtruth_batch[1][img_idx],:]
        best_IOUs = np.zeros(len(groundtruth))
        for i, object in enumerate(groundtruth):
            best_iou = 0
            for j, proposal in enumerate(proposals):
                if j == max_proposals:
                    break
                xyxy_proposal = [proposal[0], proposal[1], proposal[0]+proposal[2], proposal[1]+proposal[3]]
                iou = IOU(object,xyxy_proposal)
                if iou > best_iou:
                    best_iou = iou
            best_IOUs[i] = best_iou


        MABO.append(np.mean(best_IOUs))
        RecallRate.append(np.mean(best_IOUs>= IOU_th))
    return MABO, RecallRate

def evaluate_batch_object_proposals(groundtruth_batch, proposals_batch,IOU_th,max_proposals):
    MABO = []
    RecallRate = []
    for img_idx, proposals in enumerate(proposals_batch):
        groundtruth = groundtruth_batch[0][img_idx,:groundtruth_batch[1][img_idx],:]
        best_IOUs = np.zeros(len(groundtruth))
        for i, object in enumerate(groundtruth):
            best_iou = 0
            for j, proposal in enumerate(proposals):
                if j == max_proposals:
                    break
                xyxy_proposal = [proposal[0], proposal[1], proposal[0]+proposal[2], proposal[1]+proposal[3]]
                iou = IOU(object,xyxy_proposal)
                if iou > best_iou:
                    best_iou = iou
            best_IOUs[i] = best_iou


        MABO.append(np.mean(best_IOUs))
        RecallRate.append(np.mean(best_IOUs>= IOU_th))
    return MABO, RecallRate


def prepare_proposals_database(proposals_per_image=20, data_path='../data/Potholes/', out_path = '../data/Potholes/Proposals/'):
    image_resize = 512
    max_proposals = 2000
    trainset, valset, testset, train_loader, val_loader, test_loader = load_and_transform_dataset(val_size=1,
                                                                                                  batch_size=1,
                                                                                                  image_resize=image_resize,
                                                                                     data_path=data_path, shuffle = False)

    # for image_path in trainset.image_paths:
    #     images, (objects, num_objects) = next(iter(train_loader))
    #     proposals_per_object = np.floor(proposals_per_image / (2*num_objects))
    #     proposals_list = []
    #     proposals = get_batch_selective_search_regions(images)[0]
    #     groundtruth = objects[0, :num_objects[0]]
    #
    #     IOU_list = np.zeros((len(groundtruth),max_proposals,2))
    #     for j, proposal in enumerate(proposals):
    #         if j == max_proposals:
    #             break
    #         xyxy_proposal = [proposal[0], proposal[1], proposal[0] + proposal[2], proposal[1] + proposal[3]]
    #         proposals_list.append(xyxy_proposal)
    #         for i, rect_object in enumerate(groundtruth):
    #             IOU_list[i,j,0] = IOU(rect_object, xyxy_proposal)
    #             IOU_list[i, j,1] = j
    #
    #     max_iou = np.max(IOU_list[:,:,0],axis = 0).argsort()[::-1]
    #     sorted_IOU = IOU_list[:,max_iou,:]
    #
    #     label_count = np.zeros(len(groundtruth)+1)
    #     output_proposals = []
    #     output_labels = []
    #
    #     for idx in range(max_proposals):
    #         max_index = np.argmax(sorted_IOU[:,idx,0])
    #         if sorted_IOU[max_index,idx,0] > 0.5 and label_count[max_index] < proposals_per_object:
    #             output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
    #             output_labels.append(1)
    #             label_count[max_index] += 1
    #         elif sorted_IOU[max_index,idx,0] <0.3 and np.mod(idx ,3) == 0 and label_count[-1]<5:
    #             output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
    #             output_labels.append(0)
    #             label_count[-1] += 1
    #
    #
    #     for i in range(int(proposals_per_image-np.sum(label_count))):
    #         output_proposals.append(proposals_list[int(sorted_IOU[0,len(proposals)-max_proposals-1-i,1])])
    #         output_labels.append(0)
    #
    #     filename = os.path.basename(image_path)[:-4]
    #     out_directory = os.path.dirname(out_path + "/" + filename + "_proposals.pkl")
    #     os.makedirs(out_directory, exist_ok=True)
    #
    #     with open(out_path + "/" + filename + "_proposals.pkl", 'wb') as fp:
    #         pickle.dump((torch.Tensor(output_proposals),torch.Tensor(output_labels)), fp)

    # trainset = valset
    # train_loader = val_loader
    #
    # for image_path in trainset.image_paths:
    #     images, (objects, num_objects) = next(iter(train_loader))
    #     proposals_per_object = np.floor(proposals_per_image / (2*num_objects))
    #     proposals_list = []
    #     proposals = get_batch_selective_search_regions(images)[0]
    #     groundtruth = objects[0, :num_objects[0]]
    #
    #     IOU_list = np.zeros((len(groundtruth),max_proposals,2))
    #     for j, proposal in enumerate(proposals):
    #         if j == max_proposals:
    #             break
    #         xyxy_proposal = [proposal[0], proposal[1], proposal[0] + proposal[2], proposal[1] + proposal[3]]
    #         proposals_list.append(xyxy_proposal)
    #         for i, rect_object in enumerate(groundtruth):
    #             IOU_list[i,j,0] = IOU(rect_object, xyxy_proposal)
    #             IOU_list[i, j,1] = j
    #
    #     max_iou = np.max(IOU_list[:,:,0],axis = 0).argsort()[::-1]
    #     sorted_IOU = IOU_list[:,max_iou,:]
    #
    #     label_count = np.zeros(len(groundtruth)+1)
    #     output_proposals = []
    #     output_labels = []
    #
    #     for idx in range(max_proposals):
    #         max_index = np.argmax(sorted_IOU[:,idx,0])
    #         if sorted_IOU[max_index,idx,0] > 0.5 and label_count[max_index] < proposals_per_object:
    #             output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
    #             output_labels.append(1)
    #             label_count[max_index] += 1
    #         elif sorted_IOU[max_index,idx,0] <0.3 and np.mod(idx ,3) == 0 and label_count[-1]<5:
    #             output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
    #             output_labels.append(0)
    #             label_count[-1] += 1
    #
    #
    #     for i in range(int(proposals_per_image-np.sum(label_count))):
    #         output_proposals.append(proposals_list[int(sorted_IOU[0,len(proposals)-max_proposals-1-i,1])])
    #         output_labels.append(0)
    #
    #     filename = os.path.basename(image_path)[:-4]
    #     out_directory = os.path.dirname(out_path + "/" + filename + "_proposals.pkl")
    #     os.makedirs(out_directory, exist_ok=True)
    #
    #     with open(out_path + "/" + filename + "_proposals.pkl", 'wb') as fp:
    #         pickle.dump((torch.Tensor(output_proposals),torch.Tensor(output_labels)), fp)

    trainset = testset
    train_loader = test_loader

    for image_path in trainset.image_paths:
        images, (objects, num_objects) = next(iter(train_loader))
        proposals_per_object = np.floor(proposals_per_image / (2*num_objects))
        proposals_list = []
        proposals = get_batch_selective_search_regions(images)[0]
        groundtruth = objects[0, :num_objects[0]]

        IOU_list = np.zeros((len(groundtruth),max_proposals,2))
        for j, proposal in enumerate(proposals):
            if j == max_proposals:
                break
            xyxy_proposal = [proposal[0], proposal[1], proposal[0] + proposal[2], proposal[1] + proposal[3]]
            proposals_list.append(xyxy_proposal)
            for i, rect_object in enumerate(groundtruth):
                IOU_list[i,j,0] = IOU(rect_object, xyxy_proposal)
                IOU_list[i, j,1] = j

        max_iou = np.max(IOU_list[:,:,0],axis = 0).argsort()[::-1]
        sorted_IOU = IOU_list[:,max_iou,:]

        label_count = np.zeros(len(groundtruth)+1)
        output_proposals = []
        output_labels = []

        for idx in range(max_proposals):
            max_index = np.argmax(sorted_IOU[:,idx,0])
            if sorted_IOU[max_index,idx,0] > 0.5 and label_count[max_index] < proposals_per_object:
                output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
                output_labels.append(1)
                label_count[max_index] += 1
            elif sorted_IOU[max_index,idx,0] <0.3 and np.mod(idx ,3) == 0 and label_count[-1]<5:
                output_proposals.append(proposals_list[int(sorted_IOU[max_index,idx,1])])
                output_labels.append(0)
                label_count[-1] += 1


        for i in range(int(proposals_per_image-np.sum(label_count))):
            output_proposals.append(proposals_list[int(sorted_IOU[0,len(proposals)-max_proposals-1-i,1])])
            output_labels.append(0)

        filename = os.path.basename(image_path)[:-4]
        out_directory = os.path.dirname(out_path + "/" + filename + "_proposals.pkl")
        os.makedirs(out_directory, exist_ok=True)

        with open(out_path + "/" + filename + "_proposals.pkl", 'wb') as fp:
            pickle.dump((torch.Tensor(output_proposals),torch.Tensor(output_labels)), fp)


    return

prepare_proposals_database()