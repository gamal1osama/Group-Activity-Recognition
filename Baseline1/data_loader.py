import os
import numpy as np
from utils.helper_functions import load_yaml, load_pkl
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image



dataset_root = load_yaml('config/Baseline1.yml')['dataset_root']
output_dir = load_yaml('config/Baseline1.yml')['output_dir']

annot_pickle_file = load_pkl(f'{output_dir}/annot_all.pkl')

class_mapping = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
                        'l_winpoint': 4, 'l-pass': 5, 'l-spike': 6, 'l_set':7} 


train_videos_indices = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos_indices = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos_indices = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


train_transform = transforms.Compose([
    transforms.Resize(256),            
    transforms.RandomResizedCrop(224),  
    transforms.RandomRotation(degrees=5),                   
    transforms.ToTensor(),                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        
                         std=[0.229, 0.224, 0.225]),         
])

# no augmentation
val_transform = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),       
    transforms.ToTensor(),                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        
                         std=[0.229, 0.224, 0.225]),         
])  

class Group_Activity_Dataset(Dataset):
    def __init__(self, vid_indices, transform = None):
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

                self.data.append({
                    'frame_path': frame_path,
                    'category': torch.tensor(class_mapping[category], dtype=torch.long)
                })
        print(f"Dataset initialized with {len(self.data)} samples")




    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        item = self.data[index]
        category = item['category']
        frame = cv2.imread(item['frame_path'])

        if frame is None:
            print(f"Error: Could not load image {item['frame_path']}")
            # Return a dummy image if loading fails
            frame = torch.zeros(3, 224, 224) if self.transform else np.zeros((224, 224, 3), dtype=np.uint8)
            return frame, category
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        frame = Image.fromarray(frame)
        
        if self.transform:
            frame = self.transform(frame)

        return frame, category, item['frame_path']
    

# Test the dataset class
def check_dataset_class():
    print("Testing Group_Activity_Dataset...")
    
    # Test with a small subset of video indices
    test_vid_indices = [0, 1]  # Just test with video 0
    
    # Create a simple transform for testing
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create dataset instance
    dataset = Group_Activity_Dataset(vid_indices=test_vid_indices, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test accessing first few samples
        for i in range(min(2, len(dataset))):
            try:
                frame, category, path = dataset[i]
                print(path)
                print(f"Sample {i}:")
                print(f"  Frame shape: {frame.shape if hasattr(frame, 'shape') else 'Tensor'}")
                print(f"  Category: {category} (class {category.item()})")
                
                
                plt.figure(figsize=(8, 6))
                if hasattr(frame, 'permute'):  # If it's a tensor
                    plt.imshow(frame.permute(1, 2, 0))
                else:
                    plt.imshow(frame)
                plt.title(f"Sample 0 - Category: {category.item()}")
                plt.axis('off')
                plt.show()
                    
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
        
        # Test with DataLoader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch_idx, (frames, categories, pathes) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(pathes)
            print(f"  Frames shape: {frames.shape}")
            print(f"  Categories: {categories}")
            if batch_idx >= 1:  # Just test 2 batches
                break
    else:
        print("Dataset is empty! Check the data paths.")


def data_loader(status, batch_size):
    if status == 'train':
        train_dataset = Group_Activity_Dataset(train_videos_indices, train_transform)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader
    
    elif status == 'val':
        val_dataset = Group_Activity_Dataset(val_videos_indices, val_transform)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return val_loader
    
    elif status == 'test':
        test_dataset = Group_Activity_Dataset(test_videos_indices, val_transform)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader    
    

def check_data_loader():
    print("Testing data_loader function...")

    for status in ['train', 'val', 'test']:
        print(f"\nStatus: {status}")
        loader = data_loader(status, batch_size=2)
        print(f"  Number of batches: {len(loader)}")
        for batch_idx, (frames, categories, paths) in enumerate(loader):
            print(f"  Batch {batch_idx}:")
            print(f"    Frames shape: {frames.shape}")
            print(f"    Categories: {categories}")
            print(f"    Paths: {paths}")
            # Show first image in batch
            if hasattr(frames, 'permute'):
                img = frames[0].permute(1, 2, 0).numpy()
                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.title(f"{status} - Batch {batch_idx} - Category: {categories[0].item()}")
                plt.axis('off')
                plt.show()
            if batch_idx >= 1:  # Just test 2 batches per status
                break