import torch
import torch.nn.functional as F
import pickle

from source_code.Unet_with_convLSTM import *
from source_code.config import *
from dataprocessing import ImageMaskDataset

"""
Image Segmentation and Result Saving Script

This script loads a pre-trained U-Net model to perform image segmentation on a test dataset.
The segmentation results are saved for later analysis or use.
"""


unetConvLSTM_model.load_state_dict(torch.load(path_savedweights))
unetConvLSTM_model.to(device)
unetConvLSTM_model.eval()

save_dir = 'dataloaders'
with open(os.path.join(save_dir, f"dataloader_testp{test_patient}.pkl"), "rb") as f:
    dataloader_test = pickle.load(f)


# save_dir = 'dataloaders'
# with open(os.path.join(save_dir, f"dataloader_test.pkl"), "rb") as f:
#     dataloader_test = pickle.load(f)

if __name__ == '__main__':
    predictions = []

    for i, image_batch in enumerate(dataloader_test):

        image_to_segment = image_batch['Image'].permute(0, 2, 1, 3, 4).to(device)
        ground_truth_mask = image_batch['Mask'].to(device)

        with torch.no_grad():
            model_output = unetConvLSTM_model(image_to_segment)
            pred_probability_maps = F.softmax(model_output, dim=1)
            prediction = torch.argmax(pred_probability_maps, dim=1)


            predictions.append(prediction.cpu())
            print("Patient nr ", len(predictions), "finish!")

    torch.save(predictions, saveprediction_path)
