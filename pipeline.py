import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, split="train"):
        full_df = pd.read_csv(csv_file)
        base_df = full_df.head(5000).copy()
        
        self.label_mapping = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }
        base_df['label'] = base_df['dx'].map(self.label_mapping)
        
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, val_idx = next(gss.split(base_df, groups=base_df['lesion_id']))
        
        if split == "train":
            self.df = base_df.iloc[train_idx].copy().reset_index(drop=True)
            # ADVANCED AUGMENTATION: Help the model generalize rare features
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.df = base_df.iloc[val_idx].copy().reset_index(drop=True)
            # Validation transforms must stay standard and realistic
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        path_part1 = os.path.join("HAM10000_images_part_1", f"{image_id}.jpg")
        path_part2 = os.path.join("HAM10000_images_part_2", f"{image_id}.jpg")
        
        image_path = path_part1 if os.path.exists(path_part1) else path_part2
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
            
        return image_tensor, torch.tensor(label, dtype=torch.long)

train_dataset = SkinCancerDataset("HAM10000_metadata.csv", split="train")
val_dataset = SkinCancerDataset("HAM10000_metadata.csv", split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)