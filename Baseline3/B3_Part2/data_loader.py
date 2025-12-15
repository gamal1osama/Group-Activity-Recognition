from utils.helper_functions import load_yaml, load_pkl
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torch




seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




output_dir = load_yaml('config/configs.yml')['output_dir']
dataset_root = load_yaml('config/configs.yml')['dataset_root']

annot_pickle_file = load_pkl(f"{output_dir}/annots_one_frame_per_clip.pkl")




num_classes = 8
class_mapping = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
                'l_winpoint': 4, 'l-pass': 5, 'l-spike': 6, 'l_set':7} 





train_videos_indices = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos_indices = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos_indices = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]





# transformation with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        
                         std=[0.229, 0.224, 0.225]),         
])


# no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.ToTensor(),                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        
                         std=[0.229, 0.224, 0.225]),         
])







class Baseline3Dataset(Dataset):
    def __init__(self, vid_indices, transform=None):
        self.transform = transform
        self.data = []
        for vid_id in vid_indices:

            vid_id = str(vid_id)

            if vid_id not in annot_pickle_file:
                print(f"Warning: Video ID {vid_id} not found in annotations")
                continue

            clips_dct = annot_pickle_file[vid_id]
            for clip_id in clips_dct.keys():
                category = clips_dct[clip_id]['category']
                frame_path = f'{dataset_root}/volleyball_/videos/{vid_id}/{str(clip_id)}/{str(clip_id)}.jpg'

                players_bounding_boxes = []
                for player_bounding_box in clips_dct[clip_id]['frame_boxes_dct']:
                    players_bounding_boxes.append((player_bounding_box['X'], player_bounding_box['Y'], player_bounding_box['W'], player_bounding_box['H']))

                self.data.append({
                    'frame_path': frame_path,
                    'players_bounding_boxes': players_bounding_boxes,
                    'category': torch.tensor(class_mapping[category], dtype=torch.long),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
    
        image = Image.open(item['frame_path']).convert('RGB')
        players_bounding_boxes = item['players_bounding_boxes']
        category = item['category']

        players = []
        for box in players_bounding_boxes:
            x, y, w, h = box
            player_crop = image.crop((x, y, x + w, y + h))

            if self.transform:
                player_crop = self.transform(player_crop)

            players.append(player_crop)

    
        return players, category
    

def check_dataset_class():
    print("Testing Baseline3Dataset...")
    
    # Test with a small subset of video indices
    test_vid_indices = [0, 1]  # Just test with video 0 and 1
    dataset = Baseline3Dataset(test_vid_indices)
    
    print(f"Dataset length: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        players, category = dataset[i]
        print(f"Sample {i}: Number of players: {len(players)}, Category: {category}")
        
        # Display original image separately
        original_image = Image.open(dataset.data[i]['frame_path']).convert('RGB')
        fig_orig, ax_orig = plt.subplots(figsize=(8, 6))
        ax_orig.imshow(original_image)
        ax_orig.set_title(f"Sample {i} - Original Frame - Category: {category}")
        ax_orig.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Display player crops
        num_players = len(players)
        fig, axes = plt.subplots(1, num_players, figsize=(15, 5))
        
        for j, player_crop in enumerate(players):
            axes[j].imshow(player_crop)
            axes[j].set_title(f"Player {j + 1}")
            axes[j].axis('off')
        
        plt.suptitle(f"Sample {i} - Players - Category: {category}")
        plt.tight_layout()
        plt.show()
    
    print("Baseline3Dataset test completed.")



def collate_players(batch):
    # batch: list of (players, category)
    players_batch, category_batch = zip(*batch)
    # players_batch: tuple of lists of tensors (player crops)
    # category_batch: tuple of categories
    return list(players_batch), torch.tensor(category_batch)


def data_loader(status, batch_size):
    if status == 'train':
        train_dataset = Baseline3Dataset(train_videos_indices, train_transform)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_players)
        return train_loader
    
    elif status == 'val':
        val_dataset = Baseline3Dataset(val_videos_indices, val_transform)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_players)
        return val_loader
    
    elif status == 'test':
        test_dataset = Baseline3Dataset(test_videos_indices, val_transform)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_players)
        return test_loader 




def check_data_loader():
    print("Testing data_loader function...")

    for status in ['train', 'val', 'test']:
        print(f"\nStatus: {status}")
        loader = data_loader(status, batch_size=4)
        print(f"  Number of batches: {len(loader)}")

        dataset = loader.dataset
        print(f"  Dataset length: {len(dataset)}")
        
        # Get first batch
        batch = next(iter(loader))
        players_batch, category_batch = batch
        print(f"  First batch - Players batch size: {len(players_batch)}, Categories: {category_batch}")
        
        # Plot like in check_dataset_class
        for batch_idx in range(min(2, len(players_batch))):
            players = players_batch[batch_idx]
            category = category_batch[batch_idx]
            
            num_players = len(players)
            fig, axes = plt.subplots(1, num_players, figsize=(15, 5))
            
            if num_players == 1:
                axes = [axes]
            
            for j, player_crop in enumerate(players):
                axes[j].imshow(player_crop.permute(1, 2, 0).numpy())
                axes[j].set_title(f"Player {j + 1}")
                axes[j].axis('off')
            
            plt.suptitle(f"{status.upper()} Batch {batch_idx} - Players - Category: {category}")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    check_dataset_class()
    check_data_loader()