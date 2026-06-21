import os
import torch
import gradio as gr
from torchvision import transforms
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

transformPipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluateUiInput(inputImg):
    if inputImg is None:
        return "No image uploaded."
    
    # I convert the numpy data from Gradio back into a standard PIL object
    imgObject = Image.fromarray(inputImg.astype('uint8'), 'RGB')
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
    
    return {classes[idx]: float(probabilities[idx].item()) for idx in classes}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Clinical Skin Cancer Classifier Prototype")
    
    with gr.Row():
        with gr.Column():
            imgInput = gr.Image(label="Lesion Image Target")
            submitBtn = gr.Button("Run Diagnostic Analysis", variant="primary")
        with gr.Column():
            labelOutput = gr.Label(num_top_classes=3, label="Top Classification Outputs")
            
    gr.Markdown("Disclaimer: Research application platform. Not intended as an alternative to clinical evaluation.")
    submitBtn.click(fn=evaluateUiInput, inputs=imgInput, outputs=labelOutput)

if __name__ == "__main__":
    demo.launch()
