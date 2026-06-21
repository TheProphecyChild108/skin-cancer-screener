import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pipeline import val_loader
from model import SkinCancerResNet

def evaluate_model():
    print("Loading multi-class ResNet architecture for evaluation...")
    model = SkinCancerResNet()
    model.load_state_dict(torch.load('skin_cancer_cnn_weights.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running validation over dataset batches... please wait...")
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # The 7 specific clinical classes from the HAM10000 dataset
    target_names = [
        'Melanocytic nevi (nv)', 
        'Melanoma (mel)', 
        'Benign keratosis (bkl)', 
        'Basal cell carcinoma (bcc)', 
        'Actinic keratoses (akiec)', 
        'Vascular lesions (vasc)', 
        'Dermatofibroma (df)'
    ]

    print("\n--- CLINICAL METRICS REPORT ---")
    print(classification_report(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], target_names=target_names, zero_division=0))
    
    print("\n--- RAW CONFUSION MATRIX ---")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model()