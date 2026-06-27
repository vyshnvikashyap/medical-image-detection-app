import requests

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
        print(f"✅ {filename} downloaded!")

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
