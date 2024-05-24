import os
from torch import device, cuda

# GPU Configuration
GPU_NUM = 1
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = f"{GPU_NUM}"

device = device("cuda" if cuda.is_available() else "cpu")

# data root dir
root_dir = "/home/gryvh/workspace/revma_medianus_3d"  # remote
#root_dir = "/Users/gryveronikahaga/Documents/revma_medianus_3d"  # locally

# Model Parameters
epochs = 3
batchsize = 8
num_classes=2

sequencelength=5

# Test Patients
test_patient=8
all_patients_list = ["dataloader_testp1.pkl", "dataloader_testp2.pkl", "dataloader_testp3.pkl", "dataloader_testp4.pkl",
                "dataloader_testp5.pkl", "dataloader_testp6.pkl", "dataloader_testp7.pkl", "dataloader_testp8.pkl",
                "dataloader_testp9.pkl", "dataloader_testp10.pkl"]

# Paths to load parameters from
path_savedweights = 'saved_weights/model_weights.pt'  # Path to load saved model weights from
load_prediction_path = f"Results/model_predictions.pt"  # Path to load predictions from

# Paths to save parameters from
saveprediction_path = f"Results/model_predictions.pt"  # Path to save predictions to
saveweights_filepath = "saved_weights/model_weights.pt"  # Path to save model weights to
saveloss_filepath = "saved_losses/model_losses.npy"  # Path to save training loss values


#Loading segmentations to compare

pred1_path = "Results/fixcode.pt"
pred2_path = "Results/fixcode.pt"




