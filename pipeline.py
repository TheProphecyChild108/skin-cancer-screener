import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from PIL import Image

def getPatientSplits(metadataPath):
    # I need to ensure patients do not overlap between train and validation sets
    df = pd.read_csv(metadataPath)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainIdx, valIdx = next(gss.split(df, groups=df['patient_id']))
    
    trainDf = df.iloc[trainIdx].reset_index(drop=True)
    valDf = df.iloc[valIdx].reset_index(drop=True)
    return trainDf, valDf

class SkinDataset(Dataset):
    def __init__(self, dataframe, imageDir, transform=None):
        self.df = dataframe
        self.imageDir = imageDir
        self.transform = transform
        
        # I map the text codes to numerical indices for the loss function
        self.labelMap = {
            'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgName = self.df.iloc[idx]['image_id'] + '.jpg'
        imgPath = os.path.join(self.imageDir, imgName)
        
        image = Image.open(imgPath).convert('RGB')
        rawLabel = self.df.iloc[idx]['dx']
        label = self.labelMap[rawLabel]

        if self.transform:
            image = self.transform(image)

        return image, label
