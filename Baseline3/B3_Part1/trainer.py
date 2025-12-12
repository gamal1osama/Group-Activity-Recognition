from model import PersonActivityModel, num_classes
from data_loader import PersonDataset, train_videos_indices, data_loader, output_dir, class_mapping
import torch
from torch import nn, optim
from collections import Counter
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns





seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = PersonActivityModel(num_classes, freeze_strategy='moderate')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


class_weights = None
def weighted_classes():
    global class_weights
    train_dataset = PersonDataset(train_videos_indices)

    # Calculate class weights 
    label_counts = Counter([label.item() for _, label in tqdm(train_dataset, desc="Counting class labels")])
    total_samples = len(train_dataset)
    # Calculate normalized weights for the classes
    class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]



weighted_classes()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
print(class_weights) 





criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  lr=0.0001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)



def train(num_epochs = 20, batch_size = 32):
    train_loader = data_loader('train', batch_size)
    val_loader = data_loader('val', batch_size)

    train_history = {
        'loss': [],
        'accuracy': [],
        'f1': [],
    }
    val_history = {
        'loss': [],
        'accuracy': [],
        'f1': [],
    }


    print("=" * 100)
    print(f"{'TRAINING STARTED':^100}")
    print("=" * 100)
    print(f"Total Epochs: {num_epochs} | Batch Size: {batch_size} | Device: {device}")
    print("=" * 100)


    for epoch in range(num_epochs):
        print(f"\n{'─' * 100}")
        print(f"Epoch [{epoch+1:>2}/{num_epochs}]")
        print(f"{'─' * 100}")

        model.train()
        running_loss, correct, total_samples = 0.0, 0.0, 0.0

        predictions, ground_truths = [], []

        for batch_idx, (person_images, categories) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            person_images, categories = person_images.to(device), categories.to(device)

            optimizer.zero_grad()
            prediction = model(person_images)
            loss = criterion(prediction, categories)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            running_loss += loss.item()

            _, output = prediction.max(1)

            total_samples += categories.size(0)
            correct += (output == categories).sum().item()

            predictions.extend(output.cpu().numpy())
            ground_truths.extend(categories.cpu().numpy())

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total_samples * 100
        f1 = f1_score(ground_truths, predictions, average='weighted') * 100

        eval_loss, eval_accuracy, eval_f1 = evaluate(val_loader)

        train_history['loss'].append(avg_loss)
        train_history['accuracy'].append(accuracy)
        train_history['f1'].append(f1)

        val_history['loss'].append(eval_loss)
        val_history['accuracy'].append(eval_accuracy)
        val_history['f1'].append(eval_f1)


        print(f"{'TRAIN':<15} | Loss: {avg_loss:>8.4f} | Accuracy: {accuracy:>6.2f}% | F1-Score: {f1:>6.2f}%")
        print(f"{'VALIDATION':<15} | Loss: {eval_loss:>8.4f} | Accuracy: {eval_accuracy:>6.2f}% | F1-Score: {eval_f1:>6.2f}%")
    
    print("\n" + "=" * 100)
    print(f"{'TRAINING COMPLETED':^100}")
    print("=" * 100 + "\n")
    
    return train_history, val_history




def evaluate(data_loader, return_predictions=False):
    model.eval()
    running_loss, correct, total_samples = 0.0, 0.0, 0.0

    predictions, ground_truths = [], []

    with torch.no_grad():
        for person_images, categories in tqdm(data_loader, desc="Evaluating"):
            person_images, categories = person_images.to(device), categories.to(device)

            prediction = model(person_images)
            loss = criterion(prediction, categories)

            running_loss += loss.item()

            _, output = prediction.max(1)

            total_samples += categories.size(0)
            correct += (output == categories).sum().item()

            predictions.extend(output.cpu().numpy())
            ground_truths.extend(categories.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total_samples * 100
    f1 = f1_score(ground_truths, predictions, average='weighted') * 100

    if return_predictions:
        return avg_loss, accuracy, f1, predictions, ground_truths
    return avg_loss, accuracy, f1




def test(batch_size = 128):
    test_loader = data_loader('test', batch_size)

    test_loss, test_accuracy, test_f1, test_predictions, test_ground_truths = evaluate(test_loader, return_predictions=True)

    test_history = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'f1': test_f1,
        'predictions': test_predictions,
        'ground_truths': test_ground_truths,
    }

    print("\n" + "=" * 100)
    print(f"{'TEST RESULTS':^100}")
    print("=" * 100)
    print(f"Loss: {test_loss:>8.4f} | Accuracy: {test_accuracy:>6.2f}% | F1-Score: {test_f1:>6.2f}%")
    print("=" * 100 + "\n")

    return test_history




def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=f'{output_dir}/results/confusion_matrix.png'):

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







if __name__ == "__main__":

    # Train the model
    print("Starting training...")
    train_history, val_history = train(num_epochs=15, batch_size=32)
    
    # Plot training history
    plot_training_history(train_history, val_history, save_path=f'{output_dir}/results/Baseline3/B3_Part1/training_plots.png')
    
    # Test the model
    class_names = list(class_mapping.keys())
    print("\nTesting the model...")
    test_history = test(batch_size=128)

    # Plot confusion matrix
    plot_confusion_matrix(test_history['ground_truths'], test_history['predictions'], 
                         class_names=class_names, 
                         save_path=f'{output_dir}/results/Baseline3/B3_Part1/confusion_matrix_test_set.png')


    # Save the trained model
    torch.save(model.state_dict(), f'{output_dir}/results/Baseline3/B3_Part1/person_activity_model.pth')
    print("\nModel saved to 'person_activity_model.pth'")

    