import os
import requests
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cpu")

BT_CLASSES    = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CHEST_CLASSES = ['Normal', 'Pneumonia']
SKIN_CLASSES  = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
                 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
                 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_vit(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.heads.head = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.heads.head.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model

def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
        if token:
            response = session.get(url, params={"confirm": token}, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_models():
    download_file("1OEgcqNk6Ityy0EiVjn0yg-1joh3UhL4_", "vit_brain_tumor.pth")
    download_file("1xKArcGDWXCZ3KD0wCe08PA5Rj-L_C-Xh", "vit_chest_xray.pth")
    download_file("1BTFleNXWZGsrxbUWpT3nTDwdACuk6uvn", "vit_skin_cancer.pth")

    bt_vit = build_vit(4); bt_vit.load_state_dict(torch.load("vit_brain_tumor.pth", map_location=device)); bt_vit.eval()
    ch_vit = build_vit(2); ch_vit.load_state_dict(torch.load("vit_chest_xray.pth",  map_location=device)); ch_vit.eval()
    sk_vit = build_vit(9); sk_vit.load_state_dict(torch.load("vit_skin_cancer.pth", map_location=device)); sk_vit.eval()

    return {
        "Brain Tumor": {"model": bt_vit, "classes": BT_CLASSES},
        "Chest X-ray": {"model": ch_vit, "classes": CHEST_CLASSES},
        "Skin Cancer": {"model": sk_vit, "classes": SKIN_CLASSES}
    }

st.set_page_config(page_title="Medical Image Detection", page_icon="🧠", layout="wide")
st.title("🧠 Medical Image Disease Detection")
st.markdown("Using **ViT-B16** (Vision Transformer) across 3 medical imaging domains")

models_dict = load_models()

col1, col2 = st.columns([1, 2])

with col1:
    disease  = st.selectbox("Disease Type", ["Brain Tumor", "Chest X-ray", "Skin Cancer"])
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded and st.button("🔍 Predict", use_container_width=True):
        model   = models_dict[disease]["model"]
        classes = models_dict[disease]["classes"]
        tensor  = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, 1)[0].cpu().numpy()

        pred_idx   = probs.argmax()
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx] * 100
