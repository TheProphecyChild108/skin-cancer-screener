import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pipeline import getPatientSplits, SkinDataset
from model import SkinCancerResNet

def runEvaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valDf = getPatientSplits('HAM10000_metadata.csv')

    valTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valDataset = SkinDataset(valDf, 'HAM10000_images', transform=valTransform)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False)

    model = SkinCancerResNet().to(device)
    model.load_state_dict(torch.load('skin_cancer_cnn_weights.pth', map_location=device))
    model.eval()

    allPredictions = []
    allLabels = []

    with torch.no_grad():
        for images, labels in valLoader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            allPredictions.extend(predictions.cpu().numpy())
            allLabels.extend(labels.numpy())

    targetNames = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    print("Classification Matrix:")
    print(classification_report(allLabels, allPredictions, target_names=targetNames))
    
    # I generate and save a visual confusion matrix grid plot here
    cm = confusion_matrix(allLabels, allPredictions)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targetNames)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title('Lesion Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Visual matrix grid saved locally as confusion_matrix.png")

if __name__ == "__main__":
    runEvaluation()
