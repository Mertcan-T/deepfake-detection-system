import numpy as np

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
    """
    Kareler arası deepfake skor dalgalanmasını (standart sapma) hesaplar.

    Yorum kılavuzu:
      STD > 20 -> Anlık/kesintili deepfake (belirli karelerde yüksek, diğerlerinde düşük)
      STD < 5  + yüksek ort. -> Tutarlı, tam video deepfake manipülasyonu
      STD < 5  + düşük ort.  -> Tutarlı gerçek video
    """
    return float(np.std(skorlar)) if len(skorlar) >= 2 else 0.0
