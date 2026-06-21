import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pipeline import train_loader, train_dataset
from model import SkinCancerResNet

def train_model():
    print("🚀 Initializing Weighted Multi-Class Training Cycle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate inverse class weights based on our specific training split distribution
    labels = train_dataset.df['label'].values
    class_counts = np.bincount(labels, minlength=7)
    # Avoid dividing by zero if a class has zero counts in this specific subset
    class_counts = np.where(class_counts == 0, 1, class_counts) 
    
    weights = 1.0 / class_counts
    weights = weights / np.sum(weights) * 7.0 # Normalize weights
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"📊 Calculated Loss Penalties: {weights}")

    model = SkinCancerResNet().to(device)
    
    # Pass the calculated weights directly into CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        print(f"\n📢 --- EPOCH {epoch+1} SUMMARY ---")
        print(f"Average Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%\n")

    print("Deep Learning Weighted Training Complete!")
    torch.save(model.state_dict(), 'skin_cancer_cnn_weights.pth')
    print("✅ Optimized weights successfully exported to 'skin_cancer_cnn_weights.pth'!")

if __name__ == "__main__":
    train_model()