import os
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from model import SkinCancerResNet

# 1. Initialize model architecture and load weights globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerResNet()
weights_path = 'skin_cancer_cnn_weights.pth'

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
else:
    print(f"❌ Error: {weights_path} not found. Run train.py first!")

# 2. Define standard image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Core prediction wrapper function for Gradio
def predict(inp_img):
    if inp_img is None:
        return "Please upload an image."
    
    # Convert incoming numpy array/PIL image to standard RGB tensor
    image = Image.fromarray(inp_img.astype('uint8'), 'RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    classes = {
        0: "Melanocytic nevi (nv - Common Benign Mole)",
        1: "Melanoma (mel - High Risk Malignant)",
        2: "Benign keratosis-like lesions (bkl - Harmless Bumpy Spot)",
        3: "Basal cell carcinoma (bcc - Malignant)",
        4: "Actinic keratoses (akiec - Pre-Cancerous)",
        5: "Vascular lesions (vasc - Benign)",
        6: "Dermatofibroma (df - Benign Skin Growth)"
    }
    
    # Return dictionary structured specifically for Gradio's Label component
    return {classes[idx]: float(probabilities[idx].item()) for idx in classes}

# 4. Construct the Imperial-caliber Web Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Clinical Dermatological Image Classifier
        ### Imperial College London Candidate Research Prototype — Powered by ResNet50
        
        *Upload a close-up dermoscopy image (`.jpg` or `.png`) to generate an instant, 7-class differential diagnostic probability matrix.*
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Lesion Image")
            submit_btn = gr.Button("Analyze Lesion Structure", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Top Morphological Diagnostics")
            
    gr.Markdown(
        """
        ---
        ⚠️ **Clinical Disclaimer:** This application is a deep learning research prototype trained on the HAM10000 dataset pool. It does not replace formal histopathological evaluation or official clinical assessment by a board-certified dermatologist.
        """
    )
    
    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()