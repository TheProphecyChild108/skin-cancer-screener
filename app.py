import os
import torch
import gradio as gr
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from model import SkinCancerResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkinCancerResNet()
weightsPath = 'skin_cancer_cnn_weights.pth'

if os.path.exists(weightsPath):
    model.load_state_dict(torch.load(weightsPath, map_location=device))
    model.eval()
else:
    print("Model weights checkpoint file missing.")

filter_weights = models.MobileNet_V3_Small_Weights.DEFAULT
filter_model = models.mobilenet_v3_small(weights=filter_weights)
filter_model.eval()
filter_categories = filter_weights.meta["categories"]

transformPipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluateUiInput(inputImg):
    if inputImg is None:
        return "No image uploaded."
    
    imgObject = Image.fromarray(inputImg.astype('uint8'), 'RGB')
    
    filter_tensor = transformPipeline(imgObject).unsqueeze(0)
    with torch.no_grad():
        filter_outputs = filter_model(filter_tensor)
        filter_probs = torch.nn.functional.softmax(filter_outputs[0], dim=0)
        
    top_filter_idx = torch.argmax(filter_probs).item()
    detected_object = filter_categories[top_filter_idx]
    detected_confidence = filter_probs[top_filter_idx].item()
    
    blocked_keywords = ["web site", "comic book", "envelope", "packet", "screen", "monitor", "paper", "book jacket"]
    
    if any(keyword in detected_object for keyword in blocked_keywords) and detected_confidence > 0.12:
        return "Invalid Input Flagged: The system detected a '" + detected_object + "' instead of a clinical skin scan. Please upload a clear, close-up image of the lesion."
    
    imgTensor = transformPipeline(imgObject).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(imgTensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    classes = {
        0: "Melanocytic nevi (nv)",
        1: "Melanoma (mel)",
        2: "Benign keratosis-like lesions (bkl)",
        3: "Basal cell carcinoma (bcc)",
        4: "Actinic keratoses (akiec)",
        5: "Vascular lesions (vasc)",
        6: "Dermatofibroma (df)"
    }
    
    max_prob = torch.max(probabilities).item()
    
    if max_prob < 0.50:
        return "Inconclusive Scan: The image does not highly resemble a valid skin lesion or the scan confidence is too low."
        
    return {classes[idx]: float(probabilities[idx].item()) for idx in classes}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Clinical Skin Cancer Classifier Prototype")
    
    with gr.Row():
        with gr.Column():
            imgInput = gr.Image(
                label="Lesion Image Target", 
                sources=["upload", "webcam", "clipboard"]
            )
            submitBtn = gr.Button("Run Diagnostic Analysis", variant="primary")
        with gr.Column():
            labelOutput = gr.Label(num_top_classes=3, label="Top Classification Outputs")
            
    gr.Markdown("Disclaimer: Research application platform. Not intended as an alternative to clinical evaluation.")
    submitBtn.click(fn=evaluateUiInput, inputs=imgInput, outputs=labelOutput)

if __name__ == "__main__":
    demo.launch()
