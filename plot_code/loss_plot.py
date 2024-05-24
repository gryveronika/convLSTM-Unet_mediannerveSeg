import numpy as np
import matplotlib.pyplot as plt

# Load the data
standard_unet = np.load("../saved_losses/YOUR_FILE_NAME.npy")
advanced_unet= np.load("../saved_losses/YOUR_FILE_NAME.npy")

blue = "#80CDC1"
red = "#F4A582"

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

print(len(standard_unet[0]))
print(len(advanced_unet[0]))

# Plot on each subplot
ax[0].plot(standard_unet[0], color=blue, label="Train")
ax[0].plot(standard_unet[1], color=red, label="Validate")
ax[0].set_title("BinarySeg5*")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Dice loss")
ax[0].legend()

ax[1].plot(advanced_unet[0], color=blue, label="Train")
ax[1].plot(advanced_unet[1], color=red, label="Validate")
ax[1].set_title("convLSTM+U-Net")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Dice loss")
ax[1].legend()

plt.tight_layout()
plt.show()

