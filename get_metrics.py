import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os
import pickle

from source_code.Unet_with_convLSTM import *
from source_code.config import *
from dataprocessing import ImageMaskDataset



def get_metrics(model, dataloader, device, num_of_classes):
    """
    Calculate various evaluation metrics for a semantic segmentation model.

    Args:
        model (torch.nn.Module): The segmentation model.
        dataset (torch.utils.data.Dataset): The dataset to evaluate.
        device (torch.device): The device to perform computations on (e.g., CPU or GPU).
        num_classes (int): The number of classes in the segmentation task.
        current_class (int): The current class for binary classification.

    Returns:
        dict: A dictionary containing evaluation metrics per class.
    """

    metrics_per_class = {'iou': np.zeros(num_classes),
                         'precision': np.zeros(num_classes),
                         'f1': np.zeros(num_classes),
                         'recall': np.zeros(num_classes),
                         'acc': np.zeros(num_classes),
                         'dice': np.zeros(num_classes)}
    class_counts = np.zeros(num_classes)
    model.to(device)
    model.eval()

    with torch.no_grad():

        for i, image_batch in enumerate(dataloader):

            image_to_segment = image_batch['Image'].permute(0, 2, 1, 3, 4).to(device)
            ground_truth_mask = image_batch['Mask'].to(device)
            gt_reshaped = ground_truth_mask.long()

            model_output = model(image_to_segment)

            pred_probability_maps = F.softmax(model_output, dim=1)
            pred_mask = torch.argmax(pred_probability_maps, dim=1)
            pred_mask = pred_mask.unsqueeze(1)


            for cl in range(0, num_classes):
                temp_ground_truth = torch.where(gt_reshaped == cl, 1, 0)
                temp_pred_mask = torch.where(pred_mask == cl, 1, 0)

                # Calculate metrics only if the class is present in ground truth or prediction
                if temp_ground_truth.sum() > 0 or temp_pred_mask.sum() > 0:
                    tp, fp, fn, tn = smp.metrics.get_stats(temp_pred_mask.squeeze(), temp_ground_truth.squeeze(),
                                                           mode='binary')
                    # Check for all-zero conditions to avoid division by zero
                    all_tp_zero = torch.all(tp == 0)
                    all_fp_zero = torch.all(fp == 0)
                    all_fn_zero = torch.all(fn == 0)

                    tp_sum = torch.sum(tp)
                    fp_sum = torch.sum(fp)
                    fn_sum = torch.sum(fn)
                    tn_sum = torch.sum(tn)

                    # calculates metrics
                    if not (all_tp_zero and all_fp_zero):
                        batch_precision = tp_sum / (tp_sum + fp_sum)
                    else:
                        batch_precision = 0.0

                    if not (all_tp_zero and all_fn_zero):
                        batch_recall = tp_sum / (tp_sum + fn_sum)
                    else:
                        batch_recall = 0.0

                    if not (batch_recall == 0 and batch_precision == 0):
                        batch_f1_score = (2 * batch_recall * batch_precision) / (batch_recall + batch_precision)
                    else:
                        batch_f1_score = 0.0

                    batch_iou_score = tp_sum / (tp_sum + fp_sum + fn_sum)
                    batch_acc = (tn_sum + tp_sum) / (tp_sum + tn_sum + fn_sum + fp_sum)
                    batch_dice = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)

                    # Update metrics per class
                    metrics_per_class['iou'][cl] += batch_iou_score
                    metrics_per_class['precision'][cl] += batch_precision
                    metrics_per_class['f1'][cl] += batch_f1_score
                    metrics_per_class['recall'][cl] += batch_recall
                    metrics_per_class['acc'][cl] += batch_acc
                    metrics_per_class['dice'][cl] += batch_dice

                    class_counts[cl] += 1

    # Calculate mean across all batches and return
    for key in metrics_per_class:
        for cl in range(num_classes):
            if class_counts[cl] > 0:
                metrics_per_class[key][cl] = round(metrics_per_class[key][cl] / class_counts[cl], 4)

    return metrics_per_class


if __name__ == "__main__":
    save_dir = 'dataloaders'

    # OVERALL SCORES
    with open(os.path.join(save_dir, "dataloader_test.pkl"), "rb") as f:
        dataloader_test = pickle.load(f)

    unetConvLSTM_model.load_state_dict(torch.load(path_savedweights))
    unetConvLSTM_model.eval()

    scores = get_metrics(unetConvLSTM_model, dataloader_test, device, num_classes)

    print("Metrics overall dataset:")
    print("Iou ", scores["iou"])
    print("Precision", scores["precision"])
    print("f1", scores["f1"])
    print("dice", scores["dice"], np.mean(scores["dice"]))
    print("Recall", scores["recall"])
    print("acc", scores["acc"])

    # PATIENT SCORES
    print("Metrics per patient:")

    for patient in all_patients_list:
        with open(os.path.join(save_dir, patient), "rb") as f:
            dataloader_test = pickle.load(f)

        print("model: ", path_savedweights, "   patient: ", patient)
        scores = get_metrics(unetConvLSTM_model, dataloader_test, device, num_classes)

        print("Iou ", scores["iou"])
        print("Precision", scores["precision"])
        print("f1", scores["f1"])
        print("dice", scores["dice"], np.mean(scores["dice"]))
        print("Recall", scores["recall"])
        print("acc", scores["acc"])

