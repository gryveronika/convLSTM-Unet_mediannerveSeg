import torch
import torch.nn as nn
from collections import OrderedDict
from source_code.config import num_classes, sequencelength
# from config import num_classes, sequencelength
from torch import nn, cat, manual_seed, randn



# Original ConvLSTM cell as proposed by Shi et al. (2015)
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # Unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, activation, frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=X.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=X.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device=X.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


# Copyright (c) 2021 Rohit Panda
# The code below is used with permission from: https://github.com/sladewinter/ConvLSTM
class ConvLSTM_Layers(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers):
        super(ConvLSTM_Layers, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            )

            # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)

        # output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)


# The code below is my own
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=32, unet_levels=5):
        super(Unet, self).__init__()
        self.unet_levels = unet_levels
        features = init_features

        # Define encoder and decoder blocks
        self.encoder1 = Unet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Unet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if unet_levels == 6:
            self.encoder5 = Unet._block(features * 8, features * 16, name="enc5")
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottleneck = Unet._block(features * 16, features * 32, name="bottleneck")
            self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
            self.decoder5 = Unet._block(features * 32, features * 16, name="dec5")
        else:
            self.bottleneck = Unet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Unet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Unet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Unet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Unet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        pool1 = self.pool1(encoder1)
        encoder2 = self.encoder2(pool1)
        pool2 = self.pool2(encoder2)
        encoder3 = self.encoder3(pool2)
        pool3 = self.pool3(encoder3)
        encoder4 = self.encoder4(pool3)
        pool4 = self.pool4(encoder4)

        if self.unet_levels == 6:
            encoder5 = self.encoder5(pool4)
            pool5 = self.pool5(encoder5)
            bottleneck = self.bottleneck(pool5)
            upconv5 = self.upconv5(bottleneck)
            decoder5 = self.decoder5(cat([upconv5, encoder5], 1))
            upconv4 = self.upconv4(decoder5)
        else:
            bottleneck = self.bottleneck(pool4)
            upconv4 = self.upconv4(bottleneck)

        decoder4 = self.decoder4(cat([upconv4, encoder4], 1))
        upconv3 = self.upconv3(decoder4)
        decoder3 = self.decoder3(cat([upconv3, encoder3], 1))
        upconv2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(cat([upconv2, encoder2], 1))
        upconv1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(cat([upconv1, encoder1], 1))

        output = self.conv(decoder1)
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (name + "_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1)),
            (name + "_norm1", nn.BatchNorm2d(features)),
            (name + "_relu1", nn.ReLU(inplace=True)),
            (name + "_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1)),
            (name + "_norm2", nn.BatchNorm2d(features)),
            (name + "_relu2", nn.ReLU(inplace=True))
        ]))



class Unet_with_convLSTM(nn.Module):

    """
    Combined model of ConvLSTM and U-Net for spatiotemporal data processing.

    Args:
        num_channels (int): Number of input channels.
        hidden_states (int): Number of hidden states in the ConvLSTM layers.
        frame_size (tuple): Size of the input frame (height, width).
        num_layers (int): Number of ConvLSTM layers.
        out_channels (int): Number of output channels.
        init_unet_features (int): Initial number of features in the U-Net.

    Attributes:
        num_channels (int): Number of input channels.
        num_kernels (int): Number of hidden states in the ConvLSTM layers.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding size for the convolutional layers.
        activation (str): Activation function used in ConvLSTM.
        frame_size (tuple): Size of the input frame (height, width).
        num_layers (int): Number of ConvLSTM layers.
        in_channels (int): Number of input channels for the U-Net.
        out_channels (int): Number of output channels for the U-Net.
        init_features (int): Initial number of features in the U-Net.
        convlstm (ConvLSTM_Layers): ConvLSTM layers for spatiotemporal processing.
        unet (Unet): U-Net for spatial processing.

    Methods:
        forward(x): Forward pass through the combined ConvLSTM and U-Net model.

    """
    def __init__(self, num_channels, hidden_states, frame_size, num_layers, out_channels, init_unet_features):
        super(Unet_with_convLSTM, self).__init__()

        self.num_channels = num_channels
        self.num_kernels = hidden_states

        self.kernel_size = 3
        self.padding = 1
        self.activation = "tanh"
        self.frame_size = frame_size
        self.num_layers = num_layers

        self.in_channels = hidden_states + 1
        self.out_channels = out_channels
        self.init_features = init_unet_features

        self.convlstm = ConvLSTM_Layers(self.num_channels, self.num_kernels, self.kernel_size, self.padding,
                                        self.activation, self.frame_size, self.num_layers)

        self.unet = Unet(self.in_channels, self.out_channels, self.init_features)

    def forward(self, x):
        """
        Forward pass through the combined ConvLSTM and U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len, height, width).

        Returns:
            torch.Tensor: Output tensor from the U-Net with ConvLSTM features concatenated.
        """

        convlstm_output = self.convlstm(x)

        output_last = convlstm_output[:, :, sequencelength - 1, :, :]
        x_last = x[:, :, sequencelength - 1, :, :]

        output_last = output_last.squeeze(dim=2)
        x_last = x_last.squeeze(dim=2)

        concat_output = cat([output_last, x_last], dim=1)

        output = self.unet(concat_output)

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


unetConvLSTM_model = Unet_with_convLSTM(num_channels=1, hidden_states=4, frame_size=(416, 256), num_layers=2,
                                            out_channels=2, init_unet_features=32)




