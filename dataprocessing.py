import os
from skimage import io, color, segmentation
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import albumentations as A
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from torch import stack
import pickle
from matplotlib.colors import ListedColormap, Normalize
from source_code.config import sequencelength, batchsize, root_dir
import natsort


class ImageMaskDataset(Dataset):
    """
    A PyTorch Dataset class for loading and transforming sequences of medical images and corresponding masks.

    Args:
        data (dict): A dictionary where keys are patient IDs and values are tuples containing
                     lists of image paths and corresponding mask paths.
        sequence_length (int, optional): The length of the sequence to be extracted for each sample.
                                         Default is 10.
        dataset_type (str, optional): The type of dataset, either 'train' or 'test', which determines
                                      the transformations to be applied. Default is 'train'.

    Attributes:
        data (dict): The input data containing patient IDs and their corresponding image and mask paths.
        sequence_length (int): The number of consecutive images and masks to be used in each sample.
        step_size (int): The step size for extracting sequences from the dataset. Default is 1.
        transforms (albumentations.core.composition.ReplayCompose): The transformations to be applied
                                                                    to the images and masks.
    """

    def __init__(self, data, sequence_length=5, dataset_type='train'):
        self.data = data
        self.sequence_length = sequence_length
        self.step_size = 1  # Step size is equal to 1 image
        if dataset_type == 'train':
            self.transforms = A.ReplayCompose([
                A.Resize(height=416, width=256, interpolation=cv2.INTER_NEAREST, always_apply=True),
                A.HorizontalFlip(p=0.50),
                A.VerticalFlip(p=0.50),
            ])
        elif dataset_type == 'test':
            self.transforms = A.ReplayCompose([
                A.Resize(height=416, width=256, interpolation=cv2.INTER_NEAREST, always_apply=True)
            ])

    def __len__(self):
        """
        Returns the total number of sequences available in the dataset.
        """
        return sum(len(sequences[0]) - self.sequence_length + 1 for sequences in self.data.values())

    def __getitem__(self, idx):
        """
        Retrieves a sequence of images and the corresponding mask for the given index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            dict: A dictionary containing:
                - 'Patient_ID' (str): The ID of the patient.
                - 'Image' (torch.Tensor): A tensor of shape (sequence_length, C, H, W) containing
                                          the sequence of transformed images.
                - 'Mask' (torch.Tensor): A tensor of shape (1, H, W) containing the transformed mask
                                         corresponding to the fifth image in the sequence.
        """
        patient_ids = list(self.data.keys())

        current_count = 0
        for patient_id in patient_ids:
            sequences = self.data[patient_id]
            for i in range(len(sequences[0]) - self.sequence_length + 1):
                if current_count == idx:
                    # Extract the sequence of images and masks for the current index
                    image_sequence = []
                    mask_sequence = []
                    for j in range(i, i + self.sequence_length):
                        img = io.imread(sequences[0][j], plugin="simpleitk")
                        mask = io.imread(sequences[1][j], plugin="simpleitk")

                        if j == i:
                            flip_for_this_seq = self.transforms(image=img)

                        data_image = self.transforms.replay(flip_for_this_seq["replay"], image=img)
                        data_mask = self.transforms.replay(flip_for_this_seq["replay"], image=mask)

                        img2 = data_image["image"]
                        mask2 = data_mask["image"]

                        mask2 = mask2.astype(np.float32)

                        img2 = ToTensor()(img2)
                        mask2 = ToTensor()(mask2)

                        image_sequence.append(img2)
                        mask_sequence.append(mask2)

                    # Return only the fifth mask along with the patient ID
                    sample = {'Patient_ID': patient_id, 'Image': stack(image_sequence), 'Mask': mask_sequence[-1]}
                    return sample

                current_count += 1


if __name__ == '__main__':
    root_dir=root_dir

    all_paths = {}

    # collecting the paths in a dict where patient numbera are the keys
    for patient_dir in os.listdir(root_dir):
        patient_slices = []

        patient_path = os.path.join(root_dir, patient_dir)

        if os.path.isdir(patient_path):
            for recording_dir in os.listdir(patient_path):
                recording_path = os.path.join(patient_path, recording_dir)

                if os.path.isdir(recording_path):
                    slice_paths = [os.path.join(recording_path, filename) for filename in os.listdir(recording_path) if
                                   filename.endswith(".mhd")]
                    patient_slices.extend(slice_paths)

        # Append patient slices to all_paths
        all_paths[patient_dir] = patient_slices


    for patient_dir in all_paths.keys():
        slice_paths_for_patient = all_paths[patient_dir]

        # divide into mask and image paths
        ending_condition = "_gt.mhd"
        mask_paths_for_patient = [item for item in slice_paths_for_patient if item.endswith(ending_condition)]

        for item in mask_paths_for_patient:
            slice_paths_for_patient.remove(item)

        image_paths_for_patient = slice_paths_for_patient

        # change the ending so sorting is done correctly, and add the ending afterward
        old_ending = ".mhd"
        new_ending = "_im.mhd"
        image_paths_for_patient = [item.replace(old_ending, new_ending) for item in image_paths_for_patient]

        image_paths_for_patient=natsort.natsorted(image_paths_for_patient)
        mask_paths_for_patient=natsort.natsorted(mask_paths_for_patient)


        old_ending = "_im.mhd"
        new_ending = ".mhd"
        image_paths_for_patient = [item.replace(old_ending, new_ending) for item in image_paths_for_patient]

        # remove black/wrong segmentations
        indices_to_remove = []
        a = 0
        for i, item in enumerate(mask_paths_for_patient):
            mask_check = io.imread(item, plugin="simpleitk")
            if np.sum(mask_check) == 0:
                a += 1
                indices_to_remove.append(i)

        # Remove items from both lists after the loop
        for index in sorted(indices_to_remove, reverse=True):
            del mask_paths_for_patient[index]
            del image_paths_for_patient[index]
        print("Removed black", a)

        # Add back as a 2D array with [[images], [masks]]
        all_paths[patient_dir] = [image_paths_for_patient, mask_paths_for_patient]

    # Use the samee random patients as for 2d data
    train_indices_split=['003', '007', '023', '035', '044', '034', '051', '037', '042', '022', '039', '012', '014', '004',
                          '050', '013', '010', '024', '027', '015', '020', '021', '032', '002', '011', '016', '001', '026', '009', '048']
    val_indices=['018', '049', '043', '030', '008', '025', '017', '031']
    test_indices=['028', '006', '045', '033', '005', '036', '041', '019', '029', '052']


    train_patients_final = {index: all_paths[index] for index in train_indices_split}
    val_patients = {index: all_paths[index] for index in val_indices}
    test_patients = {index: all_paths[index] for index in test_indices}

    print(train_patients_final.keys())
    print(val_patients.keys())
    print(test_patients.keys())

    # Make datasets and dataloaders
    dataset_train = ImageMaskDataset(data=train_patients_final, sequence_length=sequencelength,  dataset_type='train')
    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, num_workers=2, drop_last=True, shuffle=True)

    dataset_validate = ImageMaskDataset(data=val_patients, sequence_length=sequencelength, dataset_type='train')
    dataloader_validate = DataLoader(dataset_validate, batch_size=batchsize, num_workers=2, drop_last=True)

    dataset_test = ImageMaskDataset(data=test_patients, sequence_length=sequencelength, dataset_type='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=2, drop_last=True)


    # Save the dataLoaders and datasets object using pickle
    save_dir = "dataloaders"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataloader_train.pkl"), "wb") as f:
        pickle.dump(dataloader_train, f)

    with open(os.path.join(save_dir, "dataloader_val.pkl"), "wb") as f:
        pickle.dump(dataloader_validate, f)

    with open(os.path.join(save_dir, "dataloader_test.pkl"), "wb") as f:
        pickle.dump(dataloader_test, f)

    save_dir = "datasets"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset_train.pkl"), "wb") as f:
        pickle.dump(dataset_train, f)

    with open(os.path.join(save_dir, "dataset_val.pkl"), "wb") as f:
        pickle.dump(dataset_validate, f)

    with open(os.path.join(save_dir, "dataset_test.pkl"), "wb") as f:
        pickle.dump(dataset_test, f)

   # Create dataloaders patient-wise
    test_dataloaders = {}

    for patient_id in test_indices:
        test_patients = {index: all_paths[index] for index in [patient_id]}
        dataset_test = ImageMaskDataset(data=test_patients, sequence_length=sequencelength, dataset_type='test')
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=2, drop_last=True)

        test_dataloaders[patient_id] = dataloader_test

    # Save per patient datasets and dataloaders
    save_dir = "dataloaders"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (patient_id, dataloader) in enumerate(test_dataloaders.items()):
        with open(os.path.join(save_dir, f"dataloader_testp{idx+1}.pkl"), "wb") as f:
            pickle.dump(dataloader, f)

