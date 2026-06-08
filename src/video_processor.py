import cv2
import time
import tempfile
import torch
import torch.nn.functional as F
import streamlit as st

from config import DEVICE, USE_AMP, MAX_BATCH_FACE, TEMPERATURE, KAYIP_LIMITI, UI_GUNCELLEME_FREKANSI
from utils import skora_gore_renk

def process_video(tfile_name, model, mtcnn, donusum, fake_idx, esik_degeri, frame_atlama, min_yuz_boyutu, temel_mtcnn_guven, ui_kare_alani, ui_ilerleme_bar, ui_canli_skor_metni, ui_fps_metni):
    """
    Videonun kare kare okunması, MTCNN ile yüz tespiti ve Xception41 ile deepfake analizinin
    gerçekleştirildiği ana işlem döngüsüdür.
    
    Parametreler:
    - tfile_name: Yüklenen videonun geçici dosya yolu.
    - model: Xception41 deepfake sınıflandırma modeli.
    - mtcnn: Yüz tespiti için MTCNN modeli.
    - donusum: Görüntüyü tensöre çeviren torchvision dönüşümleri.
    - fake_idx: Model çıktısındaki "Fake" (Sahte) sınıfının indeksi.
    - esik_degeri: Kullanıcının belirlediği deepfake karar eşiği.
    - frame_atlama: Videonun kaç karede bir işleneceğini belirten hız ayarı.
    - min_yuz_boyutu: Tespit edilecek en küçük yüzün piksel cinsi boyutu.
    - temel_mtcnn_guven: MTCNN için temel güven skoru eşiği.
    - ui_...: Streamlit arayüzündeki bileşenlerin referansları (grafik, bar, metinler vb.).
    """
    kamera = None
    video_kaydedici = None
    islenmis_video_yolu = None
    
    sonuc = {
        "tum_skorlar": [],
        "gercek_tespitler": [],
        "tum_kareler": [],
        "ai_islem_suresi": 0.0,
        "mtcnn_hata_sayaci": 0,
        "sistem_fps": 0.0,
        "islenmis_video_yolu": None,
        "hata": None
    }

    try:
        kamera = cv2.VideoCapture(tfile_name)
        toplam_kare  = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_orijinal = kamera.get(cv2.CAP_PROP_FPS)

        basarili, ilk_kare = kamera.read()
        if not basarili:
            sonuc["hata"] = "Video cozumlenemedi."
            return sonuc

        yukseklik_orj, genislik_orj = ilk_kare.shape[:2]
        MAX_GENISLIK = 1280
        if genislik_orj > MAX_GENISLIK:
            oran     = MAX_GENISLIK / genislik_orj
            genislik = MAX_GENISLIK
            yukseklik = int(yukseklik_orj * oran)
        else:
            genislik, yukseklik = genislik_orj, yukseklik_orj

        kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)

        islenmis_video_yolu = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        sonuc["islenmis_video_yolu"] = islenmis_video_yolu
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_kaydedici = cv2.VideoWriter(
            islenmis_video_yolu, fourcc, fps_orijinal, (genislik, yukseklik)
        )

        islenen_kare      = 0
        sistem_fps        = 0.0
        tum_skorlar       = []
        gercek_tespitler  = []
        tum_kareler       = []
        son_cizimler      = []
        son_bilinen_skor  = 0.0
        yuzsuz_kare_sayaci = 0
        ai_islem_suresi   = 0.0
        mtcnn_hata_sayaci  = 0
        baslangic_zamani  = time.time()

        while kamera.isOpened():
            if st.session_state.get("dur_analiz", False):
                break

            basarili, kare = kamera.read()
            if not basarili:
                break

            if kare.shape[1] > MAX_GENISLIK:
                kare = cv2.resize(kare, (genislik, yukseklik))

            islenen_kare += 1

            if islenen_kare % frame_atlama == 0:
                ai_start = time.time()
                kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

                try:
                    boxes, probs = mtcnn.detect(kare_rgb)
                except Exception:
                    mtcnn_hata_sayaci += 1
                    boxes, probs = None, None

                anlik_cizimler = []
                kare_en_yuksek = 0.0

                if boxes is not None:
                    yuzsuz_kare_sayaci = 0
                    if len(boxes) > MAX_BATCH_FACE:
                        boxes, probs = boxes[:MAX_BATCH_FACE], probs[:MAX_BATCH_FACE]

                    face_tensors, face_coords = [], []

                    for box, prob in zip(boxes, probs):
                        if prob is None:
                            continue

                        x1, y1, x2, y2 = [int(b) for b in box]

                        w, h = x2 - x1, y2 - y1
                        margin_x = int(w * 0.10)
                        margin_y = int(h * 0.10)

                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(genislik, x2 + margin_x)
                        y2 = min(yukseklik, y2 + margin_y)

                        if (x2 - x1) < min_yuz_boyutu or (y2 - y1) < min_yuz_boyutu:
                            continue

                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        if (genislik * 0.20 < mx < genislik * 0.80 and
                                yukseklik * 0.20 < my < yukseklik * 0.80):
                            esik_guven = temel_mtcnn_guven + 0.10
                        else:
                            esik_guven = temel_mtcnn_guven

                        if prob < min(esik_guven, 0.99):
                            continue

                        yuz = kare_rgb[y1:y2, x1:x2]
                        if yuz.size == 0:
                            continue

                        face_tensors.append(donusum(yuz))
                        face_coords.append((x1, y1, x2, y2))

                    if face_tensors:
                        batch_tensor = torch.stack(face_tensors).to(DEVICE)

                        with torch.inference_mode():
                            if USE_AMP:
                                with torch.autocast(device_type="cuda" if DEVICE.type=="cuda" else "cpu", dtype=torch.float16):
                                    cikti = model(batch_tensor)
                            else:
                                cikti = model(batch_tensor)

                            olasilik     = F.softmax(cikti / TEMPERATURE, dim=1)
                            fake_skorlari = olasilik[:, fake_idx].tolist()

                        for (x1, y1, x2, y2), skor in zip(face_coords, fake_skorlari):
                            if skor > kare_en_yuksek:
                                kare_en_yuksek = skor
                            etiket = (
                                f"SAHTE %{skor*100:.1f}"
                                if skor > esik_degeri
                                else f"GERCEK %{(1-skor)*100:.1f}"
                            )
                            anlik_cizimler.append((x1, y1, x2, y2, etiket, skor))

                        gercek_tespitler.append(kare_en_yuksek * 100)

                else:
                    yuzsuz_kare_sayaci += frame_atlama

                if kare_en_yuksek > 0.0:
                    son_bilinen_skor = kare_en_yuksek
                elif yuzsuz_kare_sayaci > KAYIP_LIMITI:
                    son_bilinen_skor = 0.0

                ai_islem_suresi += (time.time() - ai_start)
                tum_skorlar.append(son_bilinen_skor * 100)
                tum_kareler.append(islenen_kare)
                son_cizimler = anlik_cizimler

            for (x1, y1, x2, y2, etiket, skor) in son_cizimler:
                renk_bgr = skora_gore_renk(skor, esik_degeri, bgr=True)
                cv2.rectangle(kare, (x1, y1), (x2, y2), renk_bgr, 3)
                (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(kare, (x1, y1 - th - 10), (x1 + tw + 5, y1), renk_bgr, -1)
                cv2.putText(
                    kare, etiket, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

            video_kaydedici.write(kare)

            if islenen_kare % UI_GUNCELLEME_FREKANSI == 0:
                MAX_GOSTERIM = 640
                if genislik > MAX_GOSTERIM:
                    oran_g       = MAX_GOSTERIM / genislik
                    kare_gosterim = cv2.resize(kare, (MAX_GOSTERIM, int(yukseklik * oran_g)))
                else:
                    kare_gosterim = kare
                ui_kare_alani.image(
                    cv2.cvtColor(kare_gosterim, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )

            if islenen_kare % 5 == 0:
                ui_ilerleme_bar.progress(min(islenen_kare / max(toplam_kare, 1), 1.0))
                gecen_sure = time.time() - baslangic_zamani
                sistem_fps = islenen_kare / gecen_sure if gecen_sure > 0 else 0.0
                ui_fps_metni.metric("Sistem FPS", f"{sistem_fps:.1f}")
                ui_canli_skor_metni.metric("Anlik Skor", f"%{son_bilinen_skor*100:.1f}")

        sonuc["tum_skorlar"] = tum_skorlar
        sonuc["gercek_tespitler"] = gercek_tespitler
        sonuc["tum_kareler"] = tum_kareler
        sonuc["ai_islem_suresi"] = ai_islem_suresi
        sonuc["mtcnn_hata_sayaci"] = mtcnn_hata_sayaci
        sonuc["sistem_fps"] = sistem_fps

    except Exception as e:
        sonuc["hata"] = str(e)

    finally:
        if kamera is not None:
            kamera.release()
        if video_kaydedici is not None:
            video_kaydedici.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return sonuc
