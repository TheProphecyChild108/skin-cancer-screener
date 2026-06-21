import torch
import torch.nn as nn
import torchvision.models as models

class SkinCancerResNet(nn.Module):
    def __init__(self, numClasses=7):
        super(SkinCancerResNet, self).__init__()
        # I am using a pretrained ResNet50 backbone for feature extraction
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        inFeatures = self.backbone.fc.in_features
        # My final layer maps to the 7 specific lesion categories
        self.backbone.fc = nn.Linear(inFeatures, numClasses)

    def forward(self, x):
        return self.backbone(x)
