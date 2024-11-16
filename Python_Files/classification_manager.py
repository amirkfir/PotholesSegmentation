
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

            result_pair = [predicted, label, path]

            if predicted != label:
                scores["wrong"] += 1
            else:
                scores["correct"] += 1
            scores["total"] += 1 

            results.append(result_pair)

            if saved_images_count < max_images_to_save:
                img = image.squeeze().cpu().detach().numpy()
                img = np.transpose(img, (1, 2, 0))

                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.title(f'True Label: {label}, Predicted: {predicted}')
                plt.axis('off')

                output_path = os.path.join(output_dir, f"image_{saved_images_count + 1}.png")
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()

                saved_images_count += 1

    scores["ratio"] = scores["correct"] / scores["total"]
    print(scores)

    return results
            # Check if prediction is incorrect
            # if predicted != label:
            #     # Get the score for the predicted class
            #     score = output[0, predicted]
            #     # Backward pass to calculate gradients
            #     model.zero_grad()
            #     score.backward()
            #     # Get the saliency map by computing the maximum absolute gradient across color channels
            #     saliency, _ = torch.max(image.grad.data.abs(), dim=1)
            #     saliency = saliency.squeeze().cpu().numpy()
            #     # Unnormalize the image for visualization
            #     img = image.squeeze().cpu().detach().numpy()
            #     img = np.transpose(img, (1, 2, 0))  # Convert from CxHxW to HxWxC
            #     mean = np.array([0.5244, 0.4443, 0.3621])
            #     std = np.array([0.2679, 0.2620, 0.2733])
            #     img = std * img + mean  # Unnormalize
            #     img = np.clip(img, 0, 1)
            #     # Map class indices to class names
            #     class_names = ['Hotdog', 'Not Hotdog']
            #     true_class_name = class_names[label.item()]
            #     predicted_class_name = class_names[predicted.item()]
                # Plot the original image and its saliency map
                # plt.figure(figsize=(13, 7))
                # plt.subplot(1, 2, 1)
                # plt.imshow(img)
                # plt.title(f'Original Image\nTrue Label: {true_class_name}', fontsize=16)
                # plt.axis('off')
                # plt.subplot(1, 2, 2)
                # plt.imshow(saliency, cmap='hot')
                # plt.title(f'Saliency Map\nPrediction: {predicted_class_name} (Incorrect)', fontsize=16)
                # plt.axis('off')
                # plt.tight_layout(pad=3.0)  # Increase padding for more margin
                # plt.show()
                #return  # Exit after displaying the first misclassified image