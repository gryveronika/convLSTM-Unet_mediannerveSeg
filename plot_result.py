import matplotlib.pyplot as plt
import numpy as np
from torch import load
from matplotlib.colors import ListedColormap, Normalize
import pickle
from skimage import segmentation, color

from source_code.config import *
from dataprocessing import ImageMaskDataset

"""
This script visualizes binary and multiclass segmentation results against ground truth labels.
For each image in the test dataset, it loads the image, ground truth segmentation mask, and predicted segmentation masks.
It overlays the segmentation masks on the original images and displays them for visual inspection.
"""

if __name__ == '__main__':

    save_dir = 'dataloaders'
    with open(os.path.join(save_dir, f"dataloader_testp{test_patient}.pkl"), "rb") as f:
        dataloader_test = pickle.load(f)
    # with open(os.path.join(save_dir, f"dataloader_test.pkl"), "rb") as f:
    #     dataloader_test = pickle.load(f)

    predictions = load(pred1_path)
    predictions2 = load(pred2_path)

    for i, image_batch in enumerate(dataloader_test):
        im = image_batch["Image"][:, sequencelength - 1, :, :, :]
        seg = image_batch["Mask"]

        pred = predictions[i]
        pred2 = predictions2[i]

        us_image = im.numpy()
        us_image = np.squeeze(us_image)
        us_GT = seg.numpy()
        us_GT = np.squeeze(us_GT)
        us_pred = pred.numpy()
        us_pred = np.squeeze(us_pred)
        us_pred2 = pred2.numpy()
        us_pred2 = np.squeeze(us_pred2)

        # FOR BINARY SEGMENTATION COMPARATION

        result1_image = segmentation.mark_boundaries(us_image, us_GT, color=(0, 255, 0), mode='thin')
        result1_image = segmentation.mark_boundaries(result1_image, us_pred2, color=(255, 0, 0), mode='thin')
        result1_image = segmentation.mark_boundaries(result1_image, us_pred, color=(0, 0, 255), mode='thin')

        fig1 = plt.imshow(result1_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
