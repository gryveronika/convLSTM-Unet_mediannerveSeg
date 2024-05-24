import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



class DiceLoss(torch.nn.Module):
    """
    Computes the Dice Loss, which is used for evaluating the performance of image segmentation models.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for DiceLoss.

        Parameters:
        inputs (torch.Tensor): Predicted outputs from the model.
        targets (torch.Tensor): Ground truth labels.
        smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        torch.Tensor: Dice loss value.
        """
        preds = F.softmax(inputs, dim=1)
        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds + targets, dim=(2, 3))

        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - torch.mean(dice_coeff)
        return dice_loss

