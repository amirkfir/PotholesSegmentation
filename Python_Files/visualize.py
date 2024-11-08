import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def visualize_boxes(images, objects, num_objects):
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

def visualize_proposals(images, batch_rects, num_proposals=50, max_images=10):
    n = 0
    for idx, image in enumerate(images):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        rects = batch_rects[idx]

        im_out = image_bgr.copy()
        for i, (x, y, w, h) in enumerate(rects[:num_proposals]):
            cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 4)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        n+=1
        if n >= max_images:
            return