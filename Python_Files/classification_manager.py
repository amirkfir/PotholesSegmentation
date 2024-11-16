
import torch
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os


def get_classification_results(model, dataloader, device):

    results = []
    scores = {"correct": 0, "wrong": 0, "total": 0, "ratio": 0}

    output_dir = "./test_imgs"
    os.makedirs(output_dir, exist_ok=True)
    saved_images_count = 0
    max_images_to_save = 5

    model.eval()
    # Get a batch of images and labels from the dataloader
    # data_iter = iter(dataloader)
    for images, labels, paths in tqdm(dataloader):
        # Get a batch of images and labels from the dataloader
        # images, labels = next(data_iter)
        images = images.to(device)
        labels = labels.to(device)
        for index in range(images.shape[0]):
            image = images[index].unsqueeze(0)  # Add batch dimension
            label = labels[index]
            path = paths[index]
            # Set requires_grad=True to calculate gradients w.r.t. the input image
            image.requires_grad = True
            # Forward pass
            output = model(image)

            # print(f"output: {output.cpu().detach().numpy()}")

            # Get the predicted class
            probs = torch.sigmoid(output)

            # print(f"probs: {probs.cpu().detach().numpy()}")
           # preds = (probs >= 0.5).float()
            # _, predicted =  (probs >= 0.5).float()
            predicted =  (probs >= 0.5).float()

            # print(f"predicted: {predicted.cpu().numpy()}")

            predicted = predicted[0][1].cpu().numpy()
            label = label.cpu().numpy()

            # print(f"predicted: {predicted}")
            # print(f"label: {label}")

            result = [predicted, label, path]

            if predicted != label:
                scores["wrong"] += 1
            else:
                scores["correct"] += 1
            scores["total"] += 1 

            results.append(result)

            # if saved_images_count < max_images_to_save:
            #     print(result)

            #     img = image.squeeze().cpu().detach().numpy()
            #     img = np.transpose(img, (1, 2, 0))

            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(img)
            #     plt.title(f'True Label: {label}, Predicted: {predicted}')
            #     plt.axis('off')

            #     output_path = os.path.join(output_dir, f"image_{saved_images_count + 1}.png")
            #     plt.savefig(output_path, bbox_inches='tight')
            #     plt.close()

            #     saved_images_count += 1

    scores["ratio"] = scores["correct"] / scores["total"]
    print(scores)

    return results