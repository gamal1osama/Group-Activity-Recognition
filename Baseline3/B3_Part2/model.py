import torch
from torch import nn
from torchvision import models
from data_loader import num_classes, data_loader
from utils import helper_functions
import random
import numpy as np

seed = 44
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = helper_functions.load_yaml('config/configs.yml')['output_dir']

feature_extractor_path = f'{output_dir}/results/Baseline3/B3_Part1/person_activity_model.pth'





def get_feature_extractor(model_path=feature_extractor_path):
    """
    Load the pretrained ResNet50 model and return a feature extractor.
    
    Args:
        model_path: Path to the trained person activity model
        
    Returns:
        Feature extractor model (nn.Sequential) that outputs (batch_size, 2048) features
    """
    backbone = models.resnet50(weights=None)
    num_features = backbone.fc.in_features
    
    # Match the model structure: fc is a Sequential with a Linear layer
    fc = nn.Sequential(
        nn.Linear(num_features, 9),
    )
    backbone.fc = fc
    
    # Load state dict and remove 'backbone.' prefix if present
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k.replace('backbone.', '')] = v
        else:
            new_state_dict[k] = v
    
    backbone.load_state_dict(new_state_dict)
    
    # Remove the final classification layer to use as feature extractor
    feature_extractor = nn.Sequential(
        *list(backbone.children())[:-1],
        nn.Flatten(),
    )
    
    # Freeze all parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    return feature_extractor







class Baseline3Model(nn.Module):
    """
    Classifier model that takes pre-extracted features and classifies them.
    
    Args:
        input_dim: Dimension of input features (default: 2048 from ResNet50)
        num_classes: Number of output classes (default: 8 group activities)
    """
    def __init__(self, input_dim=2048, num_classes=num_classes):
        super(Baseline3Model, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.7), # was 0.5
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # was 0.3
            
            nn.Linear(512, num_classes)
        )

        # self._initialize_weights()

        
    def forward(self, frame_features):
        """
        Forward pass through the classifier.
        
        Args:
            frame_features: Aggregated frame features (batch_size, 2048)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        out = self.classifier(frame_features)
        return out  # (batch_size, num_classes)
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


def check_baseline3model_with_data():
    print("Testing Baseline3Model with real data...")

    loader = data_loader('train', batch_size=2)
    feature_extractor = get_feature_extractor()
    model = Baseline3Model()
    model.to(device)
    model.eval()

    batch = next(iter(loader))
    players_batch, categories = batch
    
    # Extract features
    batch_size, num_players, channels, height, width = players_batch.shape
    players_flat = players_batch.view(batch_size * num_players, channels, height, width)
    players_flat = players_flat.to(device)
    
    with torch.no_grad():
        player_features = feature_extractor(players_flat)  # (batch_size * 12, 2048)
        player_features = player_features.view(batch_size, num_players, -1)
        frame_features = torch.max(player_features, dim=1).values  # (batch_size, 2048)
        outputs = model(frame_features)
    
    print(f"Input batch: {len(players_batch)} frames")
    print(f"Output shape: {outputs.shape}")
    print("Output:", outputs)
    print("True categories:", categories)






if __name__ == '__main__':
    check_baseline3model_with_data()