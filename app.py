import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from huggingface_hub import hf_hub_download

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

@st.cache_resource
def load_models():
    HF_REPO = "Vyshnvi/medical-image-detection-models"
    
    bt_path  = hf_hub_download(repo_id=HF_REPO, filename="vit_brain_tumor.pth")
    ch_path  = hf_hub_download(repo_id=HF_REPO, filename="vit_chest_xray.pth")
    sk_path  = hf_hub_download(repo_id=HF_REPO, filename="vit_skin_cancer.pth")

    bt_vit = build_vit(4); bt_vit.load_state_dict(torch.load(bt_path, map_location=device)); bt_vit.eval()
    ch_vit = build_vit(2); ch_vit.load_state_dict(torch.load(ch_path, map_location=device)); ch_vit.eval()
    sk_vit = build_vit(9); sk_vit.load_state_dict(torch.load(sk_path, map_location=device)); sk_vit.eval()

    return {
        "Brain Tumor": {"model": bt_vit, "classes": BT_CLASSES},
        "Chest X-ray": {"model": ch_vit, "classes": CHEST_CLASSES},
        "Skin Cancer": {"model": sk_vit, "classes": SKIN_CLASSES}
    }

st.set_page_config(page_title="Medical Image Detection", page_icon="🧠", layout="wide")
st.title("🧠 Medical Image Disease Detection")
st.markdown("Using **ViT-B16** (Vision Transformer) with Grad-CAM explainability")

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

        st.success(f"**Prediction: {pred_class}** ({confidence:.2f}% confidence)")

        st.markdown("### Confidence Scores")
        for i, cls in enumerate(classes):
            st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")

st.markdown("---")
st.markdown("**GitHub:** [medical-image-detection](https://github.com/vyshnvikashyap/Medical-image-detection) | Built with PyTorch + Streamlit")
