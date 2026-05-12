import cv2
import os
import torch
from facenet_pytorch import MTCNN

def yuz_tespiti_ve_kayit_mtcnn(giris_videosu, cikti_videosu):
    """
    Belirtilen videoyu okur, MTCNN (Derin Ogrenme) ile yuzleri bulur 
    ve islenmis halini yeni bir video olarak kaydeder.
    """
    # 1. Dosya Kontrolu
    if not os.path.exists(giris_videosu):
        print("Hata: Video dosyasi bulunamadi. Lutfen dosya yolunu kontrol edin.")
        return

    # 2. MTCNN Yuz Tanima Modelini Yukle (PyTorch tabanli)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # keep_all=True ile karedeki tum yuzleri bulmasini sagliyoruz
    yuz_detektoru = MTCNN(keep_all=True, device=device)
    print(f"MTCNN Modeli {device} uzerinde basariyla yuklendi.")

    # 3. Videoyu Ac ve Teknik Ozelliklerini Al
    kamera = cv2.VideoCapture(giris_videosu)
    genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(kamera.get(cv2.CAP_PROP_FPS))
    toplam_kare = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Video Kaydediciyi Ayarla
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_kaydedici = cv2.VideoWriter(cikti_videosu, fourcc, fps, (genislik, yukseklik))
    
    print(f"Islem basladi... Toplam islenecek kare: {toplam_kare}")
    islenen_kare_sayisi = 0

    # 5. Videoyu Kare Kare Isleyen Ana Dongu
    while True:
        basarili_mi, kare = kamera.read()
        if not basarili_mi:
            break
            
        islenen_kare_sayisi += 1
        
        # MTCNN RGB formatinda calistigi icin BGR'den RGB'ye ceviriyoruz
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        
        # Yuzleri tespit et (Kutu koordinatlarini dondurur)
        boxes, _ = yuz_detektoru.detect(kare_rgb)

        # Eger yuz bulunduysa yesil dikdortgen ciz
        if boxes is not None:
            for box in boxes:
                # MTCNN float dondurur, cv2.rectangle integer bekler
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            bulunan_yuz_sayisi = len(boxes)
        else:
            bulunan_yuz_sayisi = 0

        # Bilgi metni ekle
        bilgi_metni = f"Kare: {islenen_kare_sayisi}/{toplam_kare} | Yuz: {bulunan_yuz_sayisi}"
        cv2.putText(kare, bilgi_metni, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 6. Ekranda goster ve dosyaya kaydet
        cv2.imshow("Deepfake Onisleme - MTCNN", kare)
        video_kaydedici.write(kare)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Islem kullanici tarafindan durduruldu.")
            break

    # 7. Hafizayi temizle
    kamera.release()
    video_kaydedici.release()
    cv2.destroyAllWindows()
    print("\nVideo isleme ve kaydetme islemi basariyla tamamlandi.")
    print(f"Cikti dosyasi: {cikti_videosu}")

if __name__ == "__main__":
    # Orijinal videonuzun yolu (Basindaki 'r' harfi Windows dizin yapisi icin kritiktir)
    video_yolu = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\videolar\Deepfake Example Presented by Senator Richard Blumenthal.mp4"
    
    # Islenmis videonun kaydedilecegi yol
    kayit_yolu = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\videolar\Deepfake_Example_islenmis_mtcnn.mp4"
    
    # Fonksiyonu cagir
    yuz_tespiti_ve_kayit_mtcnn(video_yolu, kayit_yolu)