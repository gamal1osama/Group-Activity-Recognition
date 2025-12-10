import random
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from data_loader import num_classes, data_loader, Group_Activity_Dataset, train_videos_indices, class_mapping, output_dir
from collections import Counter
import matplotlib.pyplot as plt
 


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GroupActivityModel(nn.Module):
    def __init__(self, num_classes, freeze_strategy='aggressive'):
        super(GroupActivityModel, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final FC layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Apply freezing strategy
        self.freeze_layers_strategy(freeze_strategy)
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_layers_strategy(self, strategy='aggressive'):
        if strategy == 'conservative':
            # Freeze everything except classifier
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
                
        elif strategy == 'moderate':  
            # Freeze early layers, train later layers
            for name, param in self.backbone.named_parameters():
                if 'layer1' in name or 'layer2' in name or 'bn1' in name or 'conv1' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        elif strategy == 'aggressive':
            # Freeze only layer1 
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False


model = GroupActivityModel(num_classes, freeze_strategy='aggressive')


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

# check_frozen_layers(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


class_weights = None
def weighted_classes():
    global class_weights
    train_dataset = Group_Activity_Dataset(train_videos_indices)

    # Calculate class weights 
    label_counts = Counter([label.item() for _, label, _ in train_dataset])  
    total_samples = len(train_dataset)
    # Calculate normalized weights for the 8 classes
    class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]


weighted_classes()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
print(class_weights)


criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



def train(num_epochs = 30, batch_size = 128):
    train_loader = data_loader('train', batch_size)
    val_loader = data_loader('val', batch_size)
    
    # History tracking
    train_history = {
        'loss': [],
        'accuracy': [],
        'f1': []
    }
    val_history = {
        'loss': [],
        'accuracy': [],
        'f1': []
    }
    
    print("=" * 100)
    print(f"{'TRAINING STARTED':^100}")
    print("=" * 100)
    print(f"Total Epochs: {num_epochs} | Batch Size: {batch_size} | Device: {device}")
    print("=" * 100)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total_samples = 0.0, 0.0, 0.0

        predictions = []
        ground_truth = []
        for bach_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            prediction = model(images)

            loss = criterion(prediction, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, output = prediction.max(1)
            
            total_samples += labels.size(0)
            correct += output.eq(labels).sum().item()

            ground_truth.extend(labels.cpu().numpy())
            predictions.extend(output.cpu().numpy())

        scheduler.step()

        average_loss = running_loss / len(train_loader)
        accuracy = (correct / total_samples) * 100.
        f1 = f1_score(ground_truth, predictions, average='weighted') * 100

        eval_loss, eval_acc, eval_f1 = evaluate(val_loader)
        
        # Store metrics
        train_history['loss'].append(average_loss)
        train_history['accuracy'].append(accuracy)
        train_history['f1'].append(f1)
        
        val_history['loss'].append(eval_loss)
        val_history['accuracy'].append(eval_acc)
        val_history['f1'].append(eval_f1)

        print(f"\n{'─' * 100}")
        print(f"Epoch [{epoch+1:>3}/{num_epochs}]")
        print(f"{'─' * 100}")
        print(f"{'TRAIN':<15} | Loss: {average_loss:>8.4f} | Accuracy: {accuracy:>6.2f}% | F1-Score: {f1:>6.2f}%")
        print(f"{'VALIDATION':<15} | Loss: {eval_loss:>8.4f} | Accuracy: {eval_acc:>6.2f}% | F1-Score: {eval_f1:>6.2f}%")
    
    print("\n" + "=" * 100)
    print(f"{'TRAINING COMPLETED':^100}")
    print("=" * 100 + "\n")
    
    return train_history, val_history


def evaluate(loader, return_predictions=False):
    """
    Evaluate model on given data loader.
    
    Args:
        loader: DataLoader object for evaluation
        return_predictions: If True, return predictions and ground truth
    
    Returns:
        tuple: (average_loss, accuracy, f1_score) or (average_loss, accuracy, f1_score, predictions, ground_truth)
    """
    model.eval()
    running_loss, correct, total_samples = 0.0, 0.0, 0.0
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            
            prediction = model(images)
            loss = criterion(prediction, labels)
            
            running_loss += loss.item()
            
            _, output = prediction.max(1)
            total_samples += labels.size(0)
            correct += output.eq(labels).sum().item()
            
            ground_truth.extend(labels.cpu().numpy())
            predictions.extend(output.cpu().numpy())
            
    average_loss = running_loss / len(loader)
    accuracy = (correct / total_samples) * 100.
    f1 = f1_score(ground_truth, predictions, average='weighted') * 100
    
    if return_predictions:
        return average_loss, accuracy, f1, predictions, ground_truth
    return average_loss, accuracy, f1



def test(batch_size=128, plot_cm=True, class_names=None):
    """Test the model and print results."""
    test_loader = data_loader('test', batch_size)
    test_loss, test_acc, test_f1, predictions, ground_truth = evaluate(test_loader, return_predictions=True)
    
    test_history = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    print("\n" + "=" * 100)
    print(f"{'TEST RESULTS':^100}")
    print("=" * 100)
    print(f"Loss: {test_loss:>8.4f} | Accuracy: {test_acc:>6.2f}% | F1-Score: {test_f1:>6.2f}%")
    print("=" * 100 + "\n")
    
    if plot_cm:
        plot_confusion_matrix(ground_truth, predictions, class_names=class_names, save_path='test_confusion_matrix.png')
    
    return test_history


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to '{save_path}'")
    plt.show()


def plot_training_history(train_history, val_history, save_path=f'{output_dir}/results/training_plots.png'):
    """
    Plot training and validation metrics.
    
    Args:
        train_history (dict): Dictionary with 'loss', 'accuracy', 'f1' lists for training
        val_history (dict): Dictionary with 'loss', 'accuracy', 'f1' lists for validation
        save_path (str): Path to save the plot
    """
    epochs = range(1, len(train_history['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_history['loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_history['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score plot
    axes[2].plot(epochs, train_history['f1'], 'b-', label='Train F1-Score', linewidth=2)
    axes[2].plot(epochs, val_history['f1'], 'r-', label='Validation F1-Score', linewidth=2)
    axes[2].set_title('Training and Validation F1-Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1-Score (%)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to '{save_path}'")
    plt.show()


if __name__ == '__main__':
    
    # Train the model
    print("Starting training...")
    train_history, val_history = train(num_epochs=30, batch_size=128)
    
    # Plot training history
    plot_training_history(train_history, val_history, save_path=f'{output_dir}/results/Baseline1/training_plots.png')
    
    # Test the model
    class_names = list(class_mapping.keys())
    print("\nTesting the model...")
    test_history = test(batch_size=128, plot_cm=True, class_names=class_names)

    # Save the trained model
    torch.save(model.state_dict(), f'{output_dir}/results/Baseline1/group_activity_model.pth')
    print("\nModel saved to 'group_activity_model.pth'")



