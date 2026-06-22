import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt # I added this to handle the plotting
from pipeline import getPatientSplits, SkinDataset
from model import SkinCancerResNet

def computeClassWeights(dataframe):
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

    trainTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # I also need basic transforms for the validation loop check
    valTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainDataset = SkinDataset(trainDf, 'HAM10000_images', transform=trainTransform)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)

    valDataset = SkinDataset(valDf, 'HAM10000_images', transform=valTransform)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False)

    model = SkinCancerResNet().to(device)
    weights = computeClassWeights(trainDf).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # I am tracking both historical lines now
    trainLossHistory = []
    valLossHistory = []

    for epoch in range(5):
        # Training Phase
        model.train()
        runningTrainLoss = 0.0
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningTrainLoss += loss.item()
            
        epochTrainLoss = runningTrainLoss / len(trainLoader)
        trainLossHistory.append(epochTrainLoss)

        # Validation Phase - I check performance without updating weights
        model.eval()
        runningValLoss = 0.0
        with torch.no_grad():
            for images, labels in valLoader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                runningValLoss += loss.item()

        epochValLoss = runningValLoss / len(valLoader)
        valLossHistory.append(epochValLoss)

        print("Epoch " + str(epoch+1) + " -> Train Loss: " + str(round(epochTrainLoss, 4)) + " | Val Loss: " + str(round(epochValLoss, 4)))

    torch.save(model.state_dict(), 'skin_cancer_cnn_weights.pth')
    
    # I update my plot script to draw both curves together
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), trainLossHistory, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, 6), valLossHistory, marker='s', color='r', label='Validation Loss')
    plt.title('ResNet50 Training vs Validation Convergence')
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('learning_curve.png')
    print("Updated optimization graph saved as learning_curve.png")
    
if __name__ == "__main__":
    runTraining()
