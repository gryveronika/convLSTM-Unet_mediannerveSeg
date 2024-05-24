from torch import nn


def weights_init(m):
    """
    Initializes weights of the given layer 'm'.

    - For convolutional layers ('Conv' in class name), uses Xavier uniform initialization.
    - For batch normalization layers ('BatchNorm' in class name), uses normal distribution initialization for weights (mean=1.0, std=0.2) and constant initialization for biases (0).

    Usage:
    model.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)