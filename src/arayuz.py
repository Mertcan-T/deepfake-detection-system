import streamlit as st
import os
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import tempfile

# Sayfa Yapılandırması
st.set_page_config(page_title="Deepfake Tespit Sistemi", page_icon="🛡️", layout="wide")

st.title("Yapay Zeka Tabanlı Deepfake Manipülasyon Tespiti")
st.write("Yüz videolarındaki sahtecilik ve manipülasyonları MTCNN ve InceptionResnetV1 mimarileri ile tespit edin.")

st.sidebar.header("Model Ayarları")
esik_degeri = st.sidebar.slider("Hassasiyet (Eşik Değeri)", 0.0, 1.0, 0.5, 0.05)

# Modelleri Önbelleğe Al (Her seferinde yeniden yüklenmesin)
@st.cache_resource
def modelleri_yukle():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yuz_detektoru = MTCNN(keep_all=True, device=device)
    siniflandirici = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2).eval().to(device)
    return yuz_detektoru, siniflandirici, device

yuz_detektoru, siniflandirici, device = modelleri_yukle()

# Görüntü Dönüşüm
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Video Yükleme Alanı
yuklenen_dosya = st.file_uploader("Analiz etmek istediğiniz video dosyasını yükleyin (.mp4, .avi)", type=["mp4", "avi", "mov"])

if yuklenen_dosya is not None:
    # Geçici bir dosyaya videoyu kaydet
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(yuklenen_dosya.read())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Giriş Videosu")
        st.video(yuklenen_dosya)
        
    with col2:
        st.subheader("Canlı Analiz Çıktısı")
        baslat_butonu = st.button("Deepfake Analizini Başlat")
        
    if baslat_butonu:
        kamera = cv2.VideoCapture(tfile.name)
        genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
        yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Streamlit canlı görüntü alanı
        st_kare_alani = st.image([])
        ilerleme_bar = st.progress(0)
        toplam_kare = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))
        islenen_kare = 0
        
        while kamera.isOpened():
            basarili_mi, kare = kamera.read()
            if not basarili_mi:
                break
                
            islenen_kare += 1
            kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            boxes, _ = yuz_detektoru.detect(kare_rgb)
            
            en_yuksek_fake_skoru = 0.0
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(genislik, x2), min(yukseklik, y2)
                    
                    yuz_kirpilmis = kare_rgb[y1:y2, x1:x2]
                    if yuz_kirpilmis.size == 0:
                        continue
                        
                    yuz_tensor = transform(yuz_kirpilmis).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        ciktilar = siniflandirici(yuz_tensor)
                        olasiliklar = F.softmax(ciktilar, dim=1)
                        fake_skoru = olasiliklar[0][1].item()
                        if fake_skoru > en_yuksek_fake_skoru:
                            en_yuksek_fake_skoru = fake_skoru
                    
                    if fake_skoru > esik_degeri:
                        etiket = f"FAKE: {fake_skoru*100:.1f}%"
                        renk = (255, 0, 0) # Streamlit RGB bekler (Kırmızı)
                    else:
                        etiket = f"REAL: {(1-fake_skoru)*100:.1f}%"
                        renk = (0, 255, 0) # Yeşil
                        
                    cv2.rectangle(kare_rgb, (x1, y1), (x2, y2), renk, 4)
                    cv2.putText(kare_rgb, etiket, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)
            
            # Ekrana canlı kareyi bas
            st_kare_alani.image(kare_rgb, channels="RGB")
            ilerleme_bar.progress(min(islenen_kare / toplam_kare, 1.0))
            
        kamera.release()
        st.success("Analiz başarıyla tamamlandı!")
        
        # Genel Değerlendirme Kartı
        if en_yuksek_fake_skoru > esik_degeri:
            st.error(f"DİKKAT: Bu videoda yüksek oranda manipülasyon (Deepfake) tespit edilmiştir! (En Yüksek Skor: %{en_yuksek_fake_skoru*100:.1f})")
        else:
            st.success(f"GÜVENLİ: Video üzerinde biyometrik bir tutarsızlığa rastlanmadı. İçerik orijinal olarak değerlendirildi.")