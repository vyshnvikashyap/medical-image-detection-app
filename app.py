import gdown

@st.cache_resource
def load_models():
    # Download from Google Drive
    files = {
        "vit_brain_tumor.pth":  "1OEgcqNk6Ityy0EiVjn0yg-1joh3UhL4_",
        "vit_chest_xray.pth":   "1xKArcGDWXCZ3KD0wCe08PA5Rj-L_C-Xh",
        "vit_skin_cancer.pth":  "1BTFleNXWZGsrxbUWpT3nTDwdACuk6uvn"
    }
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
            print(f"✅ {filename} downloaded!")

    bt_vit = build_vit(4); bt_vit.load_state_dict(torch.load("vit_brain_tumor.pth", map_location=device)); bt_vit.eval()
    ch_vit = build_vit(2); ch_vit.load_state_dict(torch.load("vit_chest_xray.pth",  map_location=device)); ch_vit.eval()
    sk_vit = build_vit(9); sk_vit.load_state_dict(torch.load("vit_skin_cancer.pth", map_location=device)); sk_vit.eval()

    return {
        "Brain Tumor": {"model": bt_vit, "classes": BT_CLASSES},
        "Chest X-ray": {"model": ch_vit, "classes": CHEST_CLASSES},
        "Skin Cancer": {"model": sk_vit, "classes": SKIN_CLASSES}
    }
