import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from torchvision.models import (efficientnet_b3, EfficientNet_B3_Weights,
                                vit_b_16, ViT_B_16_Weights, ResNet50_Weights)

# ── Page config
st.set_page_config(page_title="Medical Image Detection", page_icon="🧠", layout="wide")

device = torch.device("cpu")

# ── Classes
BT_CLASSES    = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CHEST_CLASSES = ['Normal', 'Pneumonia']
SKIN_CLASSES  = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
                 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
                 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

# ── Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Model builders
def build_resnet50(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return model

def build_efficientnet(num_classes):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return model

def build_vit(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.heads.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.heads.head.in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return model

# ── Load model
@st.cache_resource
def load_models():
    bt_r  = build_resnet50(4);    bt_r.load_state_dict(torch.load("models/resnet50_brain_tumor.pth",      map_location=device)); bt_r.eval()
    bt_e  = build_efficientnet(4); bt_e.load_state_dict(torch.load("models/efficientnet_brain_tumor.pth",  map_location=device)); bt_e.eval()
    bt_v  = build_vit(4);          bt_v.load_state_dict(torch.load("models/vit_brain_tumor.pth",           map_location=device)); bt_v.eval()
    sk_r  = build_resnet50(9);    sk_r.load_state_dict(torch.load("models/resnet50_skin_cancer.pth",      map_location=device)); sk_r.eval()
    sk_e  = build_efficientnet(9); sk_e.load_state_dict(torch.load("models/efficientnet_skin_cancer.pth",  map_location=device)); sk_e.eval()
    sk_v  = build_vit(9);          sk_v.load_state_dict(torch.load("models/vit_skin_cancer.pth",           map_location=device)); sk_v.eval()
    ch_r  = build_resnet50(2);    ch_r.load_state_dict(torch.load("models/resnet50_chest_xray.pth",       map_location=device)); ch_r.eval()
    ch_e  = build_efficientnet(2); ch_e.load_state_dict(torch.load("models/efficientnet_chest_xray.pth",   map_location=device)); ch_e.eval()
    ch_v  = build_vit(2);          ch_v.load_state_dict(torch.load("models/vit_chest_xray.pth",            map_location=device)); ch_v.eval()
    return {"Brain Tumor":  {"ResNet50": bt_r, "EfficientNetB3": bt_e, "ViT-B16": bt_v, "classes": BT_CLASSES},
            "Chest X-ray":  {"ResNet50": ch_r, "EfficientNetB3": ch_e, "ViT-B16": ch_v, "classes": CHEST_CLASSES},
            "Skin Cancer":  {"ResNet50": sk_r, "EfficientNetB3": sk_e, "ViT-B16": sk_v, "classes": SKIN_CLASSES}}

# ── GradCAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.gradients = None; self.activations = None
        for p in target_layer.parameters(): p.requires_grad_(True)
        self.hf = target_layer.register_forward_hook(lambda m,i,o: setattr(self, 'activations', o.detach()))
        self.hb = target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, tensor, class_idx):
        with torch.enable_grad():
            out = self.model(tensor)
            self.model.zero_grad()
            out[0, class_idx].backward()
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        self.hf.remove(); self.hb.remove()
        return cam

# ── UI
st.title("🧠 Medical Image Disease Detection")
st.markdown("Compare **ResNet50**, **EfficientNetB3**, and **ViT-B16** with **Grad-CAM** explainability")

models_dict = load_models()

col1, col2 = st.columns([1, 2])

with col1:
    disease  = st.selectbox("Disease Type", ["Brain Tumor", "Chest X-ray", "Skin Cancer"])
    arch     = st.selectbox("Model", ["ResNet50", "EfficientNetB3", "ViT-B16"])
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded and st.button("🔍 Predict", use_container_width=True):
        model   = models_dict[disease][arch]
        classes = models_dict[disease]["classes"]
        tensor  = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, 1)[0].cpu().numpy()

        pred_idx   = probs.argmax()
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx] * 100

        st.success(f"**Prediction: {pred_class}** ({confidence:.2f}% confidence)")

        # Confidence bar chart
        st.markdown("### Confidence Scores")
        for i, cls in enumerate(classes):
            st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")

        # GradCAM
        if arch in ["ResNet50", "EfficientNetB3"]:
            st.markdown("### Grad-CAM Heatmap")
            target = model.layer4[-1].conv3 if arch == "ResNet50" else model.features[-1][0]
            gradcam = GradCAM(model, target)
            cam = gradcam.generate(tensor, pred_idx)
            cam = cv2.resize(cam, (224, 224))
            heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            img_arr = np.array(img.resize((224, 224)))
            overlay = (0.5*img_arr + 0.5*heatmap).astype(np.uint8)
            st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
        else:
            st.info("Grad-CAM not available for ViT — attention maps coming soon!")

st.markdown("---")
st.markdown("**GitHub:** [medical-image-detection](https://github.com/vyshnvikashyap/Medical-image-detection) | Built with PyTorch + Streamlit")
