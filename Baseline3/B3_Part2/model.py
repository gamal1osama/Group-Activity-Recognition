import torch
from torch import nn
from torchvision import models
from data_loader import num_classes, data_loader
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



feature_extractor_path = r'/media/gamal/D/materials/DL/DL projects/Group Activity Recognition/results/Baseline3/B3_Part1/person_activity_model.pth'









class Baseline3Model(nn.Module):
    def __init__(self, num_classes=num_classes, model_path=feature_extractor_path):
        super(Baseline3Model, self).__init__()
        self.feature_extractor = self._load_feature_extractor(model_path)

        # New classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, players_batch):
        # players_batch: (batch_size, num_players, 3, H, W) or list of [num_players x 3 x H x W] tensors
        features = []
        with torch.no_grad():
            for players in players_batch:  # players: list of (3, H, W) tensors
                if isinstance(players, list):
                    # Stack list of player tensors into a single tensor
                    players = torch.stack(players).to(device)
                else:
                    players = players.to(device)
                feat = self.feature_extractor(players)  # feat shape: (num_players, 2048)
                agg = feat.max(dim=0).values  # agg shape: (2048,)
                features.append(agg)

            features = torch.stack(features)  # shape: (batch_size, 2048)

        # features: (batch_size, 2048)
        out = self.classifier(features)

        return out  # shape: (batch_size, num_classes)
    

    def _load_feature_extractor(self, model_path):

        self.backbone = models.resnet50(weights=None)
        
        num_features = self.backbone.fc.in_features

        # Replace the final classifier with the custom FC layers
        self.fc = nn.Sequential(
            nn.Linear(num_features, 9),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, num_classes),
        )

        # Replace the original fc layer with the custom fc
        self.backbone.fc = self.fc

        # Load state dict and remove 'backbone.' prefix
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_state_dict[k.replace('backbone.', '')] = v
        
        self.backbone.load_state_dict(new_state_dict)

        # Remove the final classification layer to use as feature extractor
        feature_extractor_model = nn.Sequential(
            *list(self.backbone.children())[:-1],
            nn.Flatten(),
        )

        for param in feature_extractor_model.parameters():
            param.requires_grad = False

        feature_extractor_model.to(device)
        feature_extractor_model.eval()

        return feature_extractor_model



def check_baseline3model_with_data():
    print("Testing Baseline3Model with real data...")

    loader = data_loader('train', batch_size=2)
    model = Baseline3Model()
    model.to(device)
    model.eval()

    batch = next(iter(loader))
    players_batch, categories = batch

    with torch.no_grad():
        outputs = model(players_batch)
    
    print(f"Input batch: {len(players_batch)} frames")
    print(f"Output shape: {outputs.shape}")
    print("Output:", outputs)
    print("True categories:", categories)



if __name__ == '__main__':
    check_baseline3model_with_data()