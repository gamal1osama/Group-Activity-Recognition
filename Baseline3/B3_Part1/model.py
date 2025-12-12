import torch 
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from data_loader import num_classes
import random
import numpy as np





seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class PersonActivityModel(nn.Module):
    def __init__(self, num_classes=num_classes, freeze_strategy=None):
        super(PersonActivityModel, self).__init__()
        # Load the pre-trained ResNet50 model
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        num_features = self.backbone.fc.in_features

       
        # Replace the final classifier with the custom FC layers
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, num_classes),
        )

        # Replace the original fc layer with the custom fc
        self.backbone.fc = self.fc

        if freeze_strategy:
            self.freeze_layers_strategy(strategy=freeze_strategy)
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_layers_strategy(self, strategy):
        if strategy == 'conservative':
            # Freeze everything except classifier
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
              
        elif strategy == 'moderate':  
            # Freeze early layers, train later layers
            for name, param in self.backbone.named_parameters():
                if 'layer1' in name or 'layer2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        elif strategy == 'aggressive':
            # Freeze only layer1 
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False

    

def check_frozen_layers(model):
    print("Frozen vs Trainable Parameters:")
    total_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if not param.requires_grad:
            frozen_params += param.numel()
            print(f"FROZEN: {name}")
        else:
            print(f"TRAINABLE: {name}")
    
    print(f"\nSummary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params*100:.1f}%)")



if __name__ == "__main__":
    
    model = PersonActivityModel(freeze_strategy='moderate')
    check_frozen_layers(model)  
