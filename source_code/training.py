import numpy as np
from torch import save, no_grad, where, tensor
import torch.nn.functional as F


def train_segmentation(dataloader_train, dataloader_validate, epochs, model, device, optimizer, num_classes, criterion,
          saveweight_filepath):
    """
    Train a segmentation model.

    Args:
        dataloader_train (DataLoader): DataLoader for training dataset.
        dataloader_validate (DataLoader): DataLoader for validation dataset.
        epochs (int): Number of epochs for training.
        model (torch.nn.Module): Segmentation model to be trained.
        device (torch.device): Device to be used for training (e.g., 'cuda' for GPU or 'cpu' for CPU).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        num_classes (int): Number of classes for segmentation.
        criterion (torch.nn.Module): Loss function for optimization.
        saveweight_filepath (str): Filepath to save the best model weights.

    Returns:
        tuple: A tuple containing the training and validation losses.
    """

    print("Start training")

    patience = 20
    counter = 0
    bestloss = float("inf")

    trainloss = []
    validateloss = []

    for epoch in range(epochs):

        losses = []

        model.to(device)
        model.train()

        for i, image_batch in enumerate(dataloader_train):
            print("Epoch: ", epoch + 1, " of ", epochs, " Batch: ", i + 1, " of ", len(dataloader_train))

            image_to_segment = image_batch['Image'].permute(0, 2, 1, 3, 4).to(device)
            ground_truth_mask = image_batch['Mask'].to(device)

            ground_truth_mask = ground_truth_mask.long()
            gt_onehot = F.one_hot(ground_truth_mask, num_classes=num_classes).squeeze()
            target_one_hot = gt_onehot.permute(0, 3, 1, 2).float()

            model_output = model(image_to_segment)

            loss = criterion(model_output, target_one_hot)

            # updates the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu())

        trainloss.append(np.mean(losses))
        print("TRAINLOSS", trainloss)

        model.eval()
        losses = []

        for i, image_batch in enumerate(dataloader_validate):
            print("Epoch: ", epoch + 1, " of ", epochs, " Validate batch: ", i + 1, " of ", len(dataloader_validate))

            image_to_segment = image_batch['Image'].permute(0, 2, 1, 3, 4).to(device)
            ground_truth_mask = image_batch['Mask'].to(device)

            gt_onehot = F.one_hot(ground_truth_mask.long(), num_classes=num_classes).squeeze()
            target_one_hot = gt_onehot.permute(0, 3, 1, 2).float()

            model_output = model(image_to_segment)

            loss = criterion(model_output, target_one_hot)
            losses.append(loss.detach().cpu())

        validateloss.append(np.mean(losses))
        print("VALIDATELOSS", validateloss)

        if validateloss[epoch] < bestloss:
            bestloss = validateloss[epoch]
            save(model.state_dict(), saveweight_filepath)
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print("Early stopping!")
                return trainloss, validateloss

    return trainloss, validateloss