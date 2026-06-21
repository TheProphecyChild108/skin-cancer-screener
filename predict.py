import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model import SkinCancerResNet

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find image file at '{image_path}'")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    model = SkinCancerResNet()
    weights_path = 'skin_cancer_cnn_weights.pth'
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Trained weights file '{weights_path}' missing.")
        return
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    print(f"\n🔍 Analyzing image: {image_path}...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()

    classes = {
        0: "Melanocytic nevi (nv - Common Benign Mole)",
        1: "Melanoma (mel - High Risk Malignant)",
        2: "Benign keratosis-like lesions (bkl - Harmless Bumpy Spot)",
        3: "Basal cell carcinoma (bcc - Malignant)",
        4: "Actinic keratoses (akiec - Pre-Cancerous)",
        5: "Vascular lesions (vasc - Benign)",
        6: "Dermatofibroma (df - Benign Skin Growth)"
    }
    
    print("=" * 65)
    print(f"📋 TOP CLINICAL PREDICTION:\n👉 {classes[prediction]}")
    print("=" * 65)
    print("📊 FULL PROBABILITY BREAKDOWN:")
    for idx, name in classes.items():
        print(f"   - {name[:30].ljust(30)}: {probabilities[idx].item()*100:.2f}%")
    print("=" * 65)
    print("⚠️  Disclaimer: This is an AI research prototype and does not replace official clinical advice.")

if __name__ == "__main__":
    # DYNAMIC TERMINAL CHECK:
    # If the user doesn't type a filename after predict.py, remind them how to use it
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py <path_to_your_image.jpg>")
        print("Example: python predict.py test_lesion.jpg")
    else:
        # Pull the filename directly out of whatever they typed in the terminal
        target_image = sys.argv[1]
        predict_image(target_image)