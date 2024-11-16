import cv2
import numpy as np
import os
import pickle
import torch
# from jupyterlab.semver import test_set

from data_loader import load_and_transform_dataset, load_data_paths, Potholes
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

        for i in range(int(proposals_per_image-np.sum(label_count))):
            output_proposals.append(proposals_list[int(sorted_IOU[0,len(proposals)-max_proposals-1-i,1])])
            output_labels.append(0)

        filename = os.path.basename(image_path)[:-4]
        out_directory = os.path.dirname(out_path + "/" + filename + "_proposals.pkl")
        os.makedirs(out_directory, exist_ok=True)

        with open(out_path + "/" + filename + "_proposals.pkl", 'wb') as fp:
            pickle.dump((torch.Tensor(output_proposals),torch.Tensor(output_labels)), fp)


    return

def prepare_proposals_images(data_path='../data/Potholes/', out_path = '../data/Potholes/Proposals/'):
    image_resize = 512

    trainable_image_paths, trainable_objects_paths, test_image_paths, test_objects_paths = load_data_paths(data_path)

    trainset = Potholes(
        target_only_transform=None,
        image_only_transform=None,
        image_paths=trainable_image_paths,
        label_paths=trainable_objects_paths
    )

    testset = Potholes(
        target_only_transform=None,
        image_only_transform=None,
        image_paths=test_image_paths,
        label_paths=test_objects_paths
    )
    #for training set
    for index, image_path in enumerate(trainset.image_paths):
        object_path = trainset.label_paths[index]
        image = cv2.imread(image_path)
        with open(object_path, 'rb') as file:
            (objects, num_objects,orginal_shape) = pickle.load(file)
        x_ratio = image_resize/ orginal_shape[0]
        y_ratio = image_resize/ orginal_shape[1]

        proposal_list_file = out_path + os.path.basename(image_path)[:-4] + "_proposals.pkl"

        with open(proposal_list_file, 'rb') as file:
            proposals_list = pickle.load(file)

        for proposal_idx, proposal in enumerate(proposals_list[0]):
            print(proposal)
            proposal[0] = np.floor(proposal[0] / x_ratio)
            proposal[1] = np.floor(proposal[1] / y_ratio)
            proposal[2] = np.floor(proposal[2] / x_ratio)
            proposal[3] = np.floor(proposal[3] / y_ratio)

            # proposal[2] = min(proposal[2],orginal_shape[0]-1)
            # proposal[3] = min(proposal[3], orginal_shape[1] - 1)


            proposal_image = image[proposal[1].int():proposal[3].int(),proposal[0].int() :proposal[2].int(),:]

            if proposals_list[1][proposal_idx]:
                out_directory = out_path +"train/Pothols/"+ os.path.basename(image_path)[:-4] + "_proposals_"+str(proposal_idx)+ ".jpg"
            else:
                out_directory = out_path + "train/NotPothols/" + os.path.basename(image_path)[:-4]+ "_proposals_"+str(proposal_idx)+ ".jpg"
            os.makedirs(os.path.dirname(out_directory), exist_ok=True)
            try:
                success = cv2.imwrite(out_directory, proposal_image)
                if not success:
                    raise IOError(f"Failed to write image file.")
            except Exception as e:
                continue


                #for testing set
    for index, image_path in enumerate(testset.image_paths):
        object_path = trainset.label_paths[index]
        image = cv2.imread(image_path)
        with open(object_path, 'rb') as file:
            (objects, num_objects,orginal_shape) = pickle.load(file)
        x_ratio = image_resize/ orginal_shape[0]
        y_ratio = image_resize/ orginal_shape[1]

        proposal_list_file = out_path + os.path.basename(image_path)[:-4] + "_proposals.pkl"

        with open(proposal_list_file, 'rb') as file:
            proposals_list = pickle.load(file)

        for proposal_idx, proposal in enumerate(proposals_list[0]):
            proposal[0] = np.floor(proposal[0] / x_ratio)
            proposal[1] = np.floor(proposal[1] / y_ratio)
            proposal[2] = np.floor(proposal[2] / x_ratio)
            proposal[3] = np.floor(proposal[3] / y_ratio)
            proposal_image = image[proposal[0].int():proposal[2].int(),proposal[1].int() :proposal[3].int(),:]

            if proposals_list[1][proposal_idx]:
                out_directory = out_path +"test/Pothols/"+ os.path.basename(image_path)[:-4] + "_proposals_"+str(proposal_idx)+ ".jpg"
            else:
                out_directory = out_path + "test/NotPothols/" + os.path.basename(image_path)[:-4]+ "_proposals_"+str(proposal_idx)+ ".jpg"
            os.makedirs(os.path.dirname(out_directory), exist_ok=True)
            try:
                success = cv2.imwrite(out_directory, proposal_image)
                if not success:
                    raise IOError(f"Failed to write image file.")
            except Exception as e:
                continue


def generate_and_save_proposals(proposals_per_image=20, data_path='../data/Potholes/', out_path='../data/Potholes/Proposals/'):
    max_proposals = 2000

    train_image_paths, train_label_paths, test_image_paths, test_label_paths = load_data_paths(data_path)

    for dataset_type in ['train', 'test']:
        if dataset_type == 'train':
            image_paths = train_image_paths
            label_paths = train_label_paths
        else:
            image_paths = test_image_paths
            label_paths = test_label_paths

        for index, image_path in enumerate(image_paths):
            label_path = label_paths[index]

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image {image_path}")
                continue

            with open(label_path, 'rb') as file:
                objects, num_objects, _ = pickle.load(file)
                groundtruth = objects[:num_objects]

            image_tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

            proposals = get_selective_search_regions(image_tensor)
            if len(proposals) > max_proposals:
                proposals = proposals[:max_proposals]

            proposals_list = []
            for proposal in proposals:
                x, y, w, h = proposal
                proposals_list.append([x, y, x + w, y + h])
            proposals_list = np.array(proposals_list)

            iou_matrix = compute_iou_matrix(groundtruth, proposals_list)

            output_proposals, output_labels = select_proposals(
                proposals_list,
                iou_matrix,
                proposals_per_image,
                num_objects
            )

            save_proposal_images(
                output_proposals,
                output_labels,
                image,
                image_path,
                dataset_type,
                out_path
            )

def compute_iou_matrix(groundtruth, proposals):
    iou_matrix = np.zeros((len(groundtruth), len(proposals)))
    for i, gt in enumerate(groundtruth):
        for j, prop in enumerate(proposals):
            iou_matrix[i, j] = IOU(gt, prop)
    return iou_matrix

def select_proposals(proposals_list, iou_matrix, proposals_per_image, num_objects):
    max_iou_per_proposal = np.max(iou_matrix, axis=0)
    max_iou_indices = np.argmax(iou_matrix, axis=0)

    sorted_indices = np.argsort(-max_iou_per_proposal)

    label_count = np.zeros(num_objects + 1)
    output_proposals = []
    output_labels = []
    proposals_per_object = np.floor(proposals_per_image / (2 * num_objects))
    negatives_needed = 5
    idx = 0

    while len(output_proposals) < proposals_per_image and idx < len(sorted_indices):
        i = sorted_indices[idx]
        max_iou = max_iou_per_proposal[i]
        obj_idx = max_iou_indices[i]
        proposal = proposals_list[i]

        if max_iou > 0.5 and label_count[obj_idx] < proposals_per_object:
            output_proposals.append(proposal)
            output_labels.append(1)
            label_count[obj_idx] += 1
        elif max_iou < 0.3 and label_count[-1] < negatives_needed:
            output_proposals.append(proposal)
            output_labels.append(0)
            label_count[-1] += 1
        idx += 1

    # Add more negatives if needed
    while len(output_proposals) < proposals_per_image and idx < len(sorted_indices):
        i = sorted_indices[idx]
        max_iou = max_iou_per_proposal[i]
        proposal = proposals_list[i]

        if max_iou < 0.3:
            output_proposals.append(proposal)
            output_labels.append(0)
        idx += 1

    return output_proposals, output_labels

def save_proposal_images(proposals, labels, image, image_path, dataset_type, out_path):
    for idx, proposal in enumerate(proposals):
        x1 = int(np.floor(proposal[0]))
        y1 = int(np.floor(proposal[1]))
        x2 = int(np.floor(proposal[2]))
        y2 = int(np.floor(proposal[3]))

        x1, x2 = np.clip([x1, x2], 0, image.shape[1] - 1)
        y1, y2 = np.clip([y1, y2], 0, image.shape[0] - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        proposal_image = image[y1:y2, x1:x2]

        label = 'Potholes' if labels[idx] == 1 else 'NotPotholes'
        out_dir = os.path.join(out_path, dataset_type, label)
        os.makedirs(out_dir, exist_ok=True)

        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_proposal_{idx}.jpg"
        out_file = os.path.join(out_dir, filename)

        if proposal_image.size == 0:
            continue

        try:
            success = cv2.imwrite(out_file, proposal_image)
            if not success:
                raise IOError(f"Failed to write image file: {out_file}")
        except Exception as e:
            print(f"Error saving image {out_file}: {e}")
            continue