from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn

def build_backbone(model_name=None, num_classes=7, freeze=True):
    if model_name == 'resnet_50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze:
            ## Freeze stem, layer1, layer2 these 3 parts
            for params in model.parameters():
                params.requires_grad = False
            
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True

        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes)
        ## Initialization for new linear layer
        nn.init.xavier_uniform_(model.fc.weight)
        if model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, 0)
    else:
        model = []

    return model