import torch.nn as nn
from torchvision.models import mobilenet_v2

def mobilenet_v2_cifar10(pretrained=False):
    model = mobilenet_v2(pretrained=pretrained)
    # Adjust classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model

