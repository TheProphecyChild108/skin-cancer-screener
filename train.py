import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from pipeline import getPatientSplits, SkinDataset
from model import SkinCancerResNet

def computeClassWeights(dataframe):
    # I calculate inverse frequency weights to fix the heavy majority class bias
    totalSamples = len(dataframe)
    labelCounts = dataframe['dx'].value_counts()
    labelMap = {'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6}
    
    classWeights = np.zeros(7)
    for countIdx, count in labelCounts.items():
        classWeights[labelMap[countIdx]] = totalSamples / (len(labelCounts)*count)
    return torch.FloatTensor(classWeights)

def runTraining():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainDf, valDf = getPatientSplits('HAM10000_metadata.csv')

    # My training augmentations to help model generalize to rare variations
    trainTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainDataset = SkinDataset(trainDf, 'HAM10000_images', transform=trainTransform)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)

    model = SkinCancerResNet().to(device)
    weights = computeClassWeights(trainDf).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # I am using a simple 5 epoch loop for my initial training runs
    model.train()
    for epoch in range(5):
        runningLoss = 0.0
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            
        print("Epoch " + str(epoch+1) + " Completed. Average Loss: " + str(round(runningLoss/len(trainLoader), 4)))

    torch.save(model.state_dict(), 'skin_cancer_cnn_weights.pth')

if __name__ == "__main__":
    runTraining()
