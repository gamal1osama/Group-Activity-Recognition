import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
import torch
from utils.helper_functions import load_yaml, load_pkl
import cv2
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
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



dataset_root = load_yaml('config/configs.yml')['dataset_root']
output_dir = load_yaml('config/configs.yml')['output_dir']

annot_pickle_file = load_pkl(f'{output_dir}/annot_all.pkl')


num_classes = 9
class_mapping = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3,
            'spiking': 4, 'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8
        }


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




class PersonDataset(Dataset):
    def __init__(self, vid_indices, transform=None):
        """
        Args:
            data (list): List of dictionaries with keys 'frame_path' and 'category'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.transform = transform

        for vid_id in vid_indices:
            vid_id = str(vid_id)
            if vid_id not in annot_pickle_file:
                print(f"Warning: Video ID {vid_id} not found in annotations")
                continue
            clips_dct = annot_pickle_file[vid_id]
            for clip_id in clips_dct.keys():
                # frame_id = int(clip_id)
                box_list = clips_dct[clip_id]['frame_boxes_dct']
                
                frame_path = f'{dataset_root}/volleyball_/videos/{vid_id}/{str(clip_id)}/{str(clip_id)}.jpg'
                    
                for person_box in box_list:
                    category = person_box['action_class']
                    self.data.append({
                        'frame_path': frame_path,
                        'X': person_box['X'],
                        'Y': person_box['Y'],
                        'W': person_box['W'],
                        'H': person_box['H'],
                        'category': torch.tensor(class_mapping[category], dtype=torch.long),
                    })
        
        print(f"Loaded {len(self.data)} person samples from videos: {vid_indices}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        frame_path = sample['frame_path']
        X, Y, W, H = sample['X'], sample['Y'], sample['W'], sample['H']
        category = sample['category']

        # Load image
        image = cv2.imread(frame_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {frame_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Crop person box
        x1, y1, x2, y2 = X, Y, X+W, Y+H
        person_image = image.crop((x1, y1, x2, y2))

        # Apply transformations if any
        if self.transform:
            person_image = self.transform(person_image)

        return person_image, category
    
def check_person_dataset_class():
    print("Testing PersonDataset...")
    
    # Test with a small subset of video indices
    test_vid_indices = [0, 1]  # Just test with video 0 and 1
    
    # Create a simple transform for testing    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create dataset instance
    dataset = PersonDataset(vid_indices=test_vid_indices, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test accessing first few samples
        for i in range(min(3, len(dataset))):
            try:
                person_image, category = dataset[i]
                print(f"\nSample {i}:")
                print(f"  Person image shape: {person_image.shape if hasattr(person_image, 'shape') else 'Tensor'}")
                print(f"  Category: {category} (class {category.item()})")
                
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                if hasattr(person_image, 'permute'):  # If it's a tensor
                    plt.imshow(person_image.permute(1, 2, 0))
                else:
                    plt.imshow(person_image)
                plt.title(f"Sample {i} - Category: {category.item()}")
                plt.axis('off')
                plt.show()
                    
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
        
        # Test with DataLoader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        for batch_idx, (person_images, categories) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Person images shape: {person_images.shape}")
            print(f"  Categories: {categories}")
            if batch_idx >= 1:  # Just test 2 batches
                break
    else:
        print("Dataset is empty! Check the data paths.")








def data_loader(status, batch_size):
    if status == 'train':
        train_dataset = PersonDataset(train_videos_indices, train_transform)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader
    
    elif status == 'val':
        val_dataset = PersonDataset(val_videos_indices, val_transform)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return val_loader
    
    elif status == 'test':
        test_dataset = PersonDataset(test_videos_indices, val_transform)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader 




def check_data_loader():
    print("Testing data_loader function...")

    for status in ['train', 'val', 'test']:
        print(f"\nStatus: {status}")
        loader = data_loader(status, batch_size=2)

        print(f"  Number of batches: {len(loader)}")
        for batch_idx, (frames, categories) in enumerate(loader):
            print(f"  Batch {batch_idx}:")
            print(f"    Frames shape: {frames.shape}")
            print(f"    Categories: {categories}")

            if hasattr(frames, 'permute'):
                img = frames[0].permute(1, 2, 0).numpy()
                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.title(f"{status} - Batch {batch_idx} - Category: {categories[0].item()}")
                plt.axis('off')
                plt.show()
            if batch_idx >= 1:  # Just test 2 batches per status
                break    





if __name__ == "__main__":
    
    check_person_dataset_class()
    check_data_loader()

