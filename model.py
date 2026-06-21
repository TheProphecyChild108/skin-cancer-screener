import torch
import torch.nn as nn
import torchvision.models as models

class SkinCancerResNet(nn.Module):
    def __init__(self):
        super(SkinCancerResNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        
        # 7 outputs now!
        self.backbone.fc = nn.Linear(num_features, 7)
        
    def forward(self, x):
        return self.backbone(x)