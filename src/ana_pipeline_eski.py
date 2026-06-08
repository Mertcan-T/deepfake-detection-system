import cv2
import os
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

def deepfake_analiz_pipeline(giris_videosu, cikti_videosu):
    # 1. Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"İşlem {device} üzerinde yapılıyor...")

    # 2. Modelleri Yükle
    # Yüz tespiti için MTCNN
    yuz_detektoru = MTCNN(keep_all=True, device=device)
    
    # Özellik çıkarımı ve sınıflandırma için InceptionResnetV1
    # Not: Prototip için pre-trained ağırlıklar kullanıyoruz.
    siniflandirici = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2).eval().to(device)
    print("MTCNN ve InceptionResnetV1 modelleri başarıyla yüklendi.")

    # 3. Görüntü Dönüşüm (Normalizasyon - 160x160)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 4. Video Ayarları
    kamera = cv2.VideoCapture(giris_videosu)
    genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(kamera.get(cv2.CAP_PROP_FPS))
    toplam_kare = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_kaydedici = cv2.VideoWriter(cikti_videosu, fourcc, fps, (genislik, yukseklik))

    islenen_kare_sayisi = 0

    # 5. Ana İşlem Döngüsü
    while True:
        basarili_mi, kare = kamera.read()
        if not basarili_mi:
            break
            
        islenen_kare_sayisi += 1
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        
        # Yüzleri tespit et
        boxes, _ = yuz_detektoru.detect(kare_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Sınırları kontrol et
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(genislik, x2), min(yukseklik, y2)
                
                # Yüzü kırp ve tensöre çevir
                yuz_kirpilmis = kare_rgb[y1:y2, x1:x2]
                
                if yuz_kirpilmis.size == 0:
                    continue

                yuz_tensor = transform(yuz_kirpilmis).unsqueeze(0).to(device)
                
                # InceptionResnetV1 ile tahmin yap
                with torch.no_grad():
                    ciktilar = siniflandirici(yuz_tensor)
                    olasiliklar = F.softmax(ciktilar, dim=1)
                    
                    # 0: Real, 1: Fake (Örnek eşleştirme)
                    fake_skoru = olasiliklar[0][1].item()
                    
                # Skor eşiğine göre etiket belirle
                if fake_skoru > 0.50:
                    etiket = f"FAKE: {fake_skoru*100:.1f}%"
                    renk = (0, 0, 255) # Kırmızı
                else:
                    etiket = f"REAL: {(1-fake_skoru)*100:.1f}%"
                    renk = (0, 255, 0) # Yeşil

                # Ekrana çiz
                cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(kare, etiket, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)

        bilgi_metni = f"Kare: {islenen_kare_sayisi}/{toplam_kare}"
        cv2.putText(kare, bilgi_metni, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Deepfake Analiz Pipeline", kare)
        video_kaydedici.write(kare)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kamera.release()
    video_kaydedici.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    # Test yolları
    video_yolu = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\videolar\Deepfake Example Presented by Senator Richard Blumenthal.mp4"
    kayit_yolu = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\videolar\Deepfake_Pipeline_Sonuc.mp4"
    
    deepfake_analiz_pipeline(video_yolu, kayit_yolu)