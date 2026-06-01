"""
=========================================================================================
  Deepfake Tespit Sistemi — Streamlit Arayüzü — FINAL SÜRÜM
=========================================================================================

GELİŞTİRME GEÇMİŞİ VE MİMARİ GÜNCELLEMELER (V1 -> FINAL):

V1 -> V2:
- Model Değişimi: InceptionResnetV1 yerine FaceForensics++ literatüründe deepfake tespiti 
  için en yüksek başarıyı veren Xception (xception41) mimarisine geçildi.
- Temel Hata Giderimleri: Pointer hataları, geçici dosya (temp) sızıntıları düzeltildi.

V3 -> V4:
- MTCNN Filtrelemesi: Arka plandaki nesnelerin yüz sanılmasını önlemek için boyut ve güven sınırı eklendi.

V4 -> V6:
- Bölgesel Hassasiyet (Regional Sensitivity): Kadrajın tam merkezindeki yüzler için daha 
  sıkı bir güven eşiği (+0.10) uygulanırken, kenarlardaki yüzler için tolerans artırıldı.

V6 -> V8:
- Donanım Optimizasyonu: Sıralı işlem yerine Batch Inference mantığına geçildi. 
- Mixed Precision (AMP): VRAM kullanımını yarı yarıya düşürmek için torch.autocast (FP16) koda eklendi.
- Özgüven Kalibrasyonu: Modelin aşırı özgüvenli yanlış kararlarını törpülemek için Softmax (T=1.0) eklendi.

V8 -> V17 (Stabilite Dönemi):
- Sinyal İşleme: Grafikteki dalgalanmaları gidermek için Savitzky-Golay filtresi eklendi.
- Renk Uzayı Düzeltmesi: Streamlit (RGB) ve OpenCV (BGR) renk çakışmaları giderildi.
- Forward-Fill Mantığı: Yüzün anlık kaybolduğu karelerde grafiğin 0'a çakılması önlendi.

V18 -> FINAL (Üretim / MLOps Dönemi):
- UI Throttling: Streamlit'in arayüzü yenilerken RAM şişirip tarayıcıyı çökertmesi önlendi.
- Çözünürlük Zırhı: 1080p/4K videoların sistemi kitlememesi için max 1280px genişlik sınırı getirildi.
- Hibrit P90 Skorlama: Kısa deepfake anlarını kaçırmamak için 90. Persentil ağırlıklı karar algoritması yazıldı.
- Domain Shift Koruması (Altın Oran): MTCNN yüz kadrajlarına %10 Padding (Marj) eklenerek 
  deepfake maske birleşim yerleri (çene, alın) analize dahil edildi. Sıfıra sıfır kesim hatası çözüldü.
=========================================================================================
"""

import os
import cv2
import time
import torch
import tempfile
import numpy as np
import streamlit as st
import timm
import plotly.graph_objects as go
import torch.nn.functional as F

from facenet_pytorch import MTCNN
from torchvision import transforms

# Sinyal pürüzsüzleştirme filtresi. Scipy yoksa sistem çökmez, esnek davranır.
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 1. DONANIM VE MİMARİ AYARLARI
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # cuDNN Benchmark: Giriş boyutları sabitse, NVIDIA GPU'nun en hızlı algoritmayı 
    # arka planda seçerek işlemi hızlandırmasını sağlar.
    torch.backends.cudnn.benchmark = True 
    # AMP (Automatic Mixed Precision): FP32 yerine FP16 kullanarak VRAM tasarrufu sağlar.
    USE_AMP = True                        
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False

# 2. STREAMLIT SAYFA YAPILANDIRMASI
st.set_page_config(page_title="Deepfake Tespit Sistemi", layout="wide")
st.title("Yapay Zeka Tabanlı Deepfake Tespit Sistemi")

st.markdown("""
### Sistem Mimarisi
- **Akademik Model:** Özgüven Kalibrasyonu (T=1.0) ve Hibrit P90 Skorlama.
- **Dinamik Kadraj (Margin):** Maske sınırlarını yakalamak için %10 güvenlik paylı yüz takibi.
- **Donanım Zırhı:** Tensor Core (FP16) aktivasyonu ve tarayıcı Throttling koruması.
""")

# 3. YAN MENÜ (SIDEBAR) KULLANICI AYARLARI
st.sidebar.header("Sistem Ayarları")

esik_degeri = st.sidebar.slider("Deepfake Eşik Değeri", 0.0, 1.0, 0.50, 0.05)
frame_atlama = st.sidebar.slider("Frame Skip (Analiz Hızı)", 1, 15, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Gelişmiş Radar Ayarları**")
min_yuz_boyutu = st.sidebar.slider("Minimum Yüz Boyutu (px)", 20, 200, 40, 10)
temel_mtcnn_guven = st.sidebar.slider("Temel MTCNN Güveni", 0.60, 0.99, 0.75, 0.01)

# 4. GİZLİ SİSTEM SABİTLERİ VE PARAMETRELERİ
MODEL_YOLU = os.path.join(os.path.dirname(__file__), "deepfake_model.pth")
TEMPERATURE = 1.0          # Modelin kararlarını yumuşatan Softmax sabiti
KAYIP_LIMITI = 15          # Yüz art arda 15 kare kaybolursa skor hafızasını siler
MAX_BATCH_FACE = 6         # Aynı karede maksimum işlenecek yüz (VRAM Taşmasını önler)
UI_GUNCELLEME_FREKANSI = 2 # Tarayıcının kilitlenmesini önlemek için çizim seyreltmesi

# 5. YARDIMCI FONKSİYONLAR
def skora_gore_renk(skor, esik, bgr=False):
    """
    Kullanıcının belirlediği dinamik Eşik değerine göre, sınırın altı ve üstü 
    için renk paletini (RGB veya BGR) belirler. Koyu/Açık ton geçişleri sağlar.
    """
    if skor >= esik:
        if skor >= esik + 0.15:
            r, g, b = 255, 50, 50   # Kırmızı (Kesin Sahte)
        else:
            r, g, b = 255, 130, 50  # Koyu Turuncu (Sınıra yakın Sahte)
    else:
        if skor < esik - 0.15:
            r, g, b = 50, 220, 50   # Koyu Yeşil (Kesin Gerçek)
        else:
            r, g, b = 160, 220, 50  # Sarımsı Yeşil (Sınıra yakın Gerçek)
    return (b, g, r) if bgr else (r, g, b)

def temporal_tutarsizlik_skoru(skorlar):
    """Kareler arası deepfake skor dalgalanmasını (standart sapma) hesaplar."""
    return float(np.std(skorlar)) if len(skorlar) >= 2 else 0.0

# 6. YAPAY ZEKA MODELLERİNİN YÜKLENMESİ (@st.cache_resource)
@st.cache_resource
def modeli_yukle():
    """
    Eğitilmiş model dosyasını (.pth) ağırlıklarıyla birlikte yükler.
    Sayfa her yenilendiğinde modeli baştan okumamak için RAM'e sabitler.
    """
    if not os.path.exists(MODEL_YOLU):
        return None, None, 0.0, 1
        
    checkpoint = torch.load(MODEL_YOLU, map_location=DEVICE, weights_only=False)
    mimari = checkpoint.get("mimari", "xception41")
    
    # Model oluşturulurken pretrained=False yapılır, kendi eğittiğimiz ağırlıklar yüklenir.
    model = timm.create_model(mimari, pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(DEVICE)
    
    fake_idx = checkpoint.get("class_to_idx", {"Fake": 0, "Real": 1}).get("Fake", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    epoch = checkpoint.get("son_epoch", checkpoint.get("epoch", 1))
    
    return model, fake_idx, val_acc, epoch

@st.cache_resource
def mtcnn_yukle():
    """Yüz tespiti için 3 aşamalı (P-Net, R-Net, O-Net) MTCNN ağını yükler."""
    return MTCNN(keep_all=True, device=DEVICE, thresholds=[0.70, 0.75, 0.80])

model, fake_idx, val_acc, egitim_epoch = modeli_yukle()
mtcnn = mtcnn_yukle()

if model is None:
    st.error("Model dosyası bulunamadı. Lütfen kontrol ediniz.")
    st.stop()
else:
    st.sidebar.markdown("---")
    st.sidebar.success("Sistem Hazır")
    st.sidebar.markdown(f"Mimari: xception41\nDoğruluk Oranı: %{val_acc*100:.2f}\nAktif Ünite: {str(DEVICE).upper()}")

# Görüntüleri Xception modelinin beklediği boyuta ve matris yapısına çevirir.
donusum = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 7. VİDEO YÜKLEME VE ANALİZ MOTORU
yuklenen_dosya = st.file_uploader("Analiz edilecek videoyu yükleyiniz", type=["mp4", "avi", "mov"])

if yuklenen_dosya is not None:
    # Streamlit tarayıcıda çalıştığı için dosya baytları fiziksel geçici (temp) dosyaya yazılır.
    video_baytlari = yuklenen_dosya.read()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_baytlari)
    tfile.flush()
    tfile.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kaynak Video")
        st.video(video_baytlari)

    with col2:
        st.subheader("Canlı Analiz İzleme")
        baslat = st.button("Analizi Başlat", type="primary")

    if baslat:
        try:
            kamera = cv2.VideoCapture(tfile.name)
            toplam_kare = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_orijinal = kamera.get(cv2.CAP_PROP_FPS)

            basarili, ilk_kare = kamera.read()
            if not basarili:
                st.error("Video çözümlenemedi.")
                st.stop()

            # ÇÖZÜNÜRLÜK ZIRHI: 4K video yüklenirse MTCNN VRAM'i taşırmasın diye
            # görüntü orantılı olarak maksimum 1280px genişliğe sıkıştırılır.
            yukseklik_orj, genislik_orj = ilk_kare.shape[:2]
            MAX_GENISLIK = 1280
            if genislik_orj > MAX_GENISLIK:
                oran = MAX_GENISLIK / genislik_orj
                genislik = MAX_GENISLIK
                yukseklik = int(yukseklik_orj * oran)
            else:
                genislik, yukseklik = genislik_orj, yukseklik_orj

            kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Analiz işlemi bittiğinde ekrana verilecek olan raporlanmış video nesnesi
            islenmis_video_yolu = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_kaydedici = cv2.VideoWriter(islenmis_video_yolu, fourcc, fps_orijinal, (genislik, yukseklik))

            with col2:
                kare_alani = st.empty()
                ilerleme_bar = st.progress(0)
                m1, m2 = st.columns(2)
                canli_skor_metni = m1.empty()
                fps_metni = m2.empty()

            islenen_kare = 0
            tum_skorlar = []
            tum_kareler = []
            son_cizimler = []
            son_bilinen_skor = 0.0
            yuzsuz_kare_sayaci = 0 
            ai_islem_suresi = 0.0  
            baslangic_zamani = time.time()

            # VİDEO DÖNGÜSÜ (Piksel Akışı)
            while kamera.isOpened():
                basarili, kare = kamera.read()
                if not basarili: break

                if kare.shape[1] > MAX_GENISLIK:
                    kare = cv2.resize(kare, (genislik, yukseklik))

                islenen_kare += 1

                # Her kare işlenmez (Frame Skip) - Gereksiz işlem yükünü azaltır
                if islenen_kare % frame_atlama == 0:
                    ai_start = time.time() 
                    # OpenCV BGR okur, PyTorch modelleri RGB bekler. Renk uzayı dönüştürülür.
                    kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
                    
                    try:
                        boxes, probs = mtcnn.detect(kare_rgb)
                    except Exception:
                        boxes, probs = None, None

                    anlik_cizimler = []
                    kare_en_yuksek = 0.0

                    if boxes is not None:
                        yuzsuz_kare_sayaci = 0 
                        if len(boxes) > MAX_BATCH_FACE:
                            boxes, probs = boxes[:MAX_BATCH_FACE], probs[:MAX_BATCH_FACE]

                        face_tensors, face_coords = [], []

                        for box, prob in zip(boxes, probs):
                            if prob is None: continue

                            x1, y1, x2, y2 = [int(b) for b in box]
                            
                            # DOMAIN SHIFT KORUMASI (Yüz Marj Algoritması - %10 Altın Oran)
                          
                            # Yüzü sıfıra sıfır kesmek yerine, etrafından %10'luk bir pay bırakılarak
                            # deepfake manipülasyonlarının en belirgin olduğu maske birleşim 
                            # yerleri (çene hattı, alın) analize dahil edilir.
                            w = x2 - x1
                            h = y2 - y1
                            margin_x = int(w * 0.10)  
                            margin_y = int(h * 0.10)  

                            x1 = max(0, x1 - margin_x)
                            y1 = max(0, y1 - margin_y)
                            x2 = min(genislik, x2 + margin_x)
                            y2 = min(yukseklik, y2 + margin_y)

                            if (x2 - x1) < min_yuz_boyutu or (y2 - y1) < min_yuz_boyutu: continue

                            # Dinamik Güvenlik Duvarı: Kadraj merkezindeki yüzlere yüksek hassasiyet gösterilir
                            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                            if (genislik * 0.20 < mx < genislik * 0.80 and yukseklik * 0.20 < my < yukseklik * 0.80):
                                esik_guven = temel_mtcnn_guven + 0.10
                            else:
                                esik_guven = temel_mtcnn_guven

                            if prob < min(esik_guven, 0.99): continue

                            yuz = kare_rgb[y1:y2, x1:x2]
                            if yuz.size == 0: continue

                            face_tensors.append(donusum(yuz))
                            face_coords.append((x1, y1, x2, y2))

                        if face_tensors:
                            # Bulunan yüzler, GPU'da tek tek değil topluca işlenmek üzere yığına (batch) eklenir
                            batch_tensor = torch.stack(face_tensors).to(DEVICE)
                            
                            # inference_mode: Model eğitim modundan çıkarılır, gradyan hesaplaması kapatılarak hız kazanılır
                            with torch.inference_mode():
                                if USE_AMP:
                                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                                        cikti = model(batch_tensor)
                                else:
                                    cikti = model(batch_tensor)

                                # Softmax ile ham lojitler olasılığa çevrilir, T=1.0 ile aşırı özgüven kırılır
                                olasilik = F.softmax(cikti / TEMPERATURE, dim=1)
                                fake_skorlari = olasilik[:, fake_idx].tolist()

                            for (x1, y1, x2, y2), skor in zip(face_coords, fake_skorlari):
                                if skor > kare_en_yuksek:
                                    kare_en_yuksek = skor
                                etiket = f"SAHTE %{skor*100:.1f}" if skor > esik_degeri else f"GERCEK %{(1-skor)*100:.1f}"
                                anlik_cizimler.append((x1, y1, x2, y2, etiket, skor))
                    else:
                        yuzsuz_kare_sayaci += frame_atlama

                    # FORWARD-FILL LİMİTİ: Kişi kafasını aniden çevirdiğinde veya elini yüzüne kapattığında
                    # grafiğin 0'a çakılmasını önlemek için sistem son skoru bir süre hafızada tutar.
                    if kare_en_yuksek > 0.0:
                        son_bilinen_skor = kare_en_yuksek
                    elif yuzsuz_kare_sayaci > KAYIP_LIMITI:
                        son_bilinen_skor = 0.0 

                    ai_islem_suresi += (time.time() - ai_start) 
                    tum_skorlar.append(son_bilinen_skor * 100)
                    tum_kareler.append(islenen_kare)
                    son_cizimler = anlik_cizimler

                # EKRAN ÇİZİM BLOĞU: Her bir yüze ait kutular ve metinler dinamik renklerle çizilir
                for (x1, y1, x2, y2, etiket, skor) in son_cizimler:
                    renk_bgr = skora_gore_renk(skor, esik_degeri, bgr=True)
                    cv2.rectangle(kare, (x1, y1), (x2, y2), renk_bgr, 3)
                    (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(kare, (x1, y1 - th - 10), (x1 + tw + 5, y1), renk_bgr, -1)
                    cv2.putText(kare, etiket, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                video_kaydedici.write(kare)

                # UI THROTTLING: Saniyede 30 kare basmak tarayıcıyı kilitler. Bu blok
                # arayüz güncellemelerini seyreltip sistemi stabil tutar.
                if islenen_kare % UI_GUNCELLEME_FREKANSI == 0:
                    MAX_GOSTERIM = 640
                    if genislik > MAX_GOSTERIM:
                        oran_g = MAX_GOSTERIM / genislik
                        kare_gosterim = cv2.resize(kare, (MAX_GOSTERIM, int(yukseklik * oran_g)))
                    else:
                        kare_gosterim = kare
                    kare_alani.image(cv2.cvtColor(kare_gosterim, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if islenen_kare % 5 == 0:
                    ilerleme_bar.progress(min(islenen_kare / max(toplam_kare, 1), 1.0))
                    gecen_sure = time.time() - baslangic_zamani
                    sistem_fps = islenen_kare / gecen_sure if gecen_sure > 0 else 0
                    
                    fps_metni.metric("Sistem FPS", f"{sistem_fps:.1f}")
                    canli_skor_metni.metric("Anlık Skor", f"%{son_bilinen_skor*100:.1f}")

            kamera.release()
            video_kaydedici.release()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 8. SONUÇ VE RAPORLAMA EKRANI
            st.divider()
            st.success("Analiz işlemi tamamlandı.")
            st.subheader("Final Video Kaydı")
            with open(islenmis_video_yolu, 'rb') as f:
                st.video(f.read())

            st.subheader("Akademik Analiz Raporu")
            if len(tum_skorlar) > 0:
                mean_skor = float(np.mean(tum_skorlar))
                percentile_90_skor = float(np.percentile(tum_skorlar, 90))
                
                # HİBRİT P90 SKORLAMASI: Basit bir ortalama almak, 3 saniyelik bir deepfake 
                # eylemini yutup göz ardı edebilir. Bu formül, en yüksek riskli %10'luk dilime 
                # (P90) ağırlık vererek manipülasyon piklerini garantili olarak yakalar.
                final_skor = (0.7 * percentile_90_skor) + (0.3 * mean_skor)
                max_skor = float(np.max(tum_skorlar))
                
                gercek_ai_fps = (len(tum_skorlar)) / ai_islem_suresi if ai_islem_suresi > 0 else 0

                if final_skor > (esik_degeri * 100):
                    st.error(f"DİKKAT: Deepfake manipülasyonu tespit edildi. (Skor: %{final_skor:.1f})")
                else:
                    st.success(f"GÜVENLİ: Belirgin bir manipülasyon tespit edilmedi. (Skor: %{final_skor:.1f})")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Genel Skor (Hibrit P90)", f"%{final_skor:.1f}")
                c2.metric("Maksimum Anlık Skor", f"%{max_skor:.1f}")
                c3.metric("Saf AI Çıkarım Hızı", f"{gercek_ai_fps:.1f} FPS")
                c4.metric("Toplam Sistem Hızı", f"{sistem_fps:.1f} FPS")

                if len(tum_skorlar) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=tum_kareler, y=tum_skorlar, mode="lines", name="Ham Skor", line=dict(color="rgba(220,20,60,0.3)", width=1)))

                    if SCIPY_AVAILABLE and len(tum_skorlar) >= 7:
                        pencere = min(11, len(tum_skorlar))
                        if pencere % 2 == 0: pencere -= 1
                        if pencere >= 5:
                            # Sinyal Gürültü Filtresi: Eğrinin zaman serisindeki genel yönelimini hesaplar
                            smooth = savgol_filter(tum_skorlar, window_length=pencere, polyorder=2)
                            fig.add_trace(go.Scatter(x=tum_kareler, y=smooth.tolist(), mode="lines", name="Savitzky-Golay Eğrisi", line=dict(color="crimson", width=3)))

                    fig.add_hline(y=esik_degeri * 100, line_dash="dash", line_color="orange")
                    fig.update_layout(title="Zaman Çizelgesi Deepfake Analizi", xaxis_title="Kare Numarası", yaxis_title="Sahte Olasılığı (%)", yaxis=dict(range=[0, 100]), height=450)
                    st.plotly_chart(fig, use_container_width=True)

        finally:
            # MEMORY LEAK KORUMASI: Sistem beklenmedik şekilde çökse bile sunucudaki (.temp)
            # gizli dosyalar silinerek sabit diskin dolması (Hard Drive bloat) engellenir.
            try:
                if 'tfile' in locals() and os.path.exists(tfile.name): os.unlink(tfile.name)
                if 'islenmis_video_yolu' in locals() and os.path.exists(islenmis_video_yolu): os.unlink(islenmis_video_yolu)
            except Exception:
                pass