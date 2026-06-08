import os
import torch
import streamlit as st
import timm
from facenet_pytorch import MTCNN
from torchvision import transforms
from config import DEVICE, MODEL_YOLU

@st.cache_resource
def modeli_yukle():
    """
    Eğitilmiş model dosyasını (.pth) ağırlıklarıyla birlikte yükler.
    Sayfa her yenilendiğinde modeli baştan okumamak için RAM'e sabitler.
    """
    if not os.path.exists(MODEL_YOLU):
        return None, None, 0.0, 1

    checkpoint = torch.load(MODEL_YOLU, map_location=DEVICE, weights_only=False)
    mimari     = checkpoint.get("mimari", "xception41")

    model = timm.create_model(mimari, pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(DEVICE)

    fake_idx  = checkpoint.get("class_to_idx", {"Fake": 0, "Real": 1}).get("Fake", 0)
    val_acc   = checkpoint.get("val_acc", 0.0)
    epoch     = checkpoint.get("son_epoch", checkpoint.get("epoch", 1))

    return model, fake_idx, val_acc, epoch

@st.cache_resource
def mtcnn_yukle():
    """Yüz tespiti için 3 aşamalı (P-Net, R-Net, O-Net) MTCNN ağını yükler."""
    return MTCNN(keep_all=True, device=DEVICE, thresholds=[0.70, 0.75, 0.80])

# Görüntüleri Xception modelinin beklediği boyuta ve matris yapısına çevirir.
donusum = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
