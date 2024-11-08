import cv2
import numpy as np

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