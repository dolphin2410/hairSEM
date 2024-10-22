import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

BLUISHNESS_THRESHOLD = 25  # TODO: This number is experimentally obtained

def load_directory(root_dir):
    images = []
    masks = []

    files = os.listdir(root_dir + "/images")
    for file in files:
        image_path = os.path.join(root_dir + "/images/", file)
        mask_path = os.path.join(root_dir + "/segmentation-masks/", file)
        images.append(image_path)
        masks.append(mask_path)

    return images, masks

def extract_masks(file_path):
    print(file_path)
    plt.figure(figsize=(15, 15))
    mask_image = cv2.imread(file_path)
    blues = mask_image[:, :, 0].copy().astype("int16")
    greens = mask_image[:, :, 1].copy().astype("int16")
    reds = mask_image[:, :, 2].copy().astype("int16")

    bluishness = np.vectorize(lambda x: 0 if x < 0 else x)((blues - greens - reds))
    maskified = np.vectorize(lambda x: 1 if x > BLUISHNESS_THRESHOLD else 0)(bluishness)

    if __name__ == "__main__":
        # For debugging purposes
        plt.imshow(maskified)
        plt.colorbar()
        plt.show()

    return maskified

if __name__ == "__main__":
    # For debugging purposes
    images, masks = load_directory("sem_images")

    for mask in masks:
        extract_masks(mask)