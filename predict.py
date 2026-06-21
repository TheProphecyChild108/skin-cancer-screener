import sys
import torch
from torchvision import transforms
from PIL import Image
from model import SkinCancerResNet

def predictSingleImage(imagePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # I set up basic transforms matching the validation setup
    valTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(imagePath).convert('RGB')
    tensorImg = valTransform(image).unsqueeze(0).to(device)

    model = SkinCancerResNet().to(device)
    model.load_state_dict(torch.load('skin_cancer_cnn_weights.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(tensorImg)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predictedClass = torch.argmax(probabilities).item()

    classes = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    print("File Target: " + str(imagePath))
    print("Top Class Match: " + str(classes[predictedClass]))
    print("Confidence Level: " + str(round(probabilities[predictedClass].item(), 4)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predictSingleImage(sys.argv[1])
    else:
        print("Please supply an image filepath argument.")
