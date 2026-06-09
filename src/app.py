"""
Deepfake Tespit Sistemi - Ana Yönetici Modülü (app.py)

Bu dosya uygulamanın giriş noktasıdır. Streamlit arayüzünü başlatır, yan menü ayarlarını
kullanıcıdan alır, modelleri belleğe yükler ve videonun `video_processor.py` üzerinden 
işlenmesini tetikler.
"""
import streamlit as st
import tempfile
import json
import numpy as np
from datetime import datetime

from config import DEVICE, TEMPERATURE, UI_GUNCELLEME_FREKANSI
from utils import temporal_tutarsizlik_skoru
from models import modeli_yukle, mtcnn_yukle, donusum
from video_processor import process_video
from ui import render_css, render_header, plot_timeline

st.set_page_config(
    page_title="Deepfake Tespit Sistemi",
    layout="wide",
    page_icon="O"
)

render_css()
render_header()

st.sidebar.header("Sistem Ayarlari")

esik_degeri     = st.sidebar.slider("Deepfake Esik Degeri", 0.0, 1.0, 0.50, 0.05)
frame_atlama    = st.sidebar.slider("Frame Skip (Analiz Hizi)", 1, 15, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Gelismiş Radar Ayarlari**")
min_yuz_boyutu     = st.sidebar.slider("Minimum Yuz Boyutu (px)", 20, 200, 40, 10)
temel_mtcnn_guven  = st.sidebar.slider("Temel MTCNN Guveni", 0.60, 0.99, 0.75, 0.01)

if "dur_analiz" not in st.session_state:
    st.session_state["dur_analiz"] = False

model, fake_idx, val_acc, egitim_epoch = modeli_yukle()
mtcnn = mtcnn_yukle()

if model is None:
    st.error("Model dosyasi bulunamadi. `src/deepfake_model.pth` konumunu kontrol ediniz.")
    st.stop()
else:
    st.sidebar.markdown("---")
    st.sidebar.success("Sistem Hazir")
    st.sidebar.markdown(
        f"**Mimari:** xception41<br>"
        f"**Dogruluk Orani:** %{val_acc*100:.2f}<br>"
        f"**Egitim Epoch:** {egitim_epoch}<br>"
        f"**Aktif Unite:** {str(DEVICE).upper()}<br>"
        f"**Temperature (T):** {TEMPERATURE}",
        unsafe_allow_html=True
    )

yuklenen_dosya = st.file_uploader(
    "Analiz edilecek videoyu yukleyiniz",
    type=["mp4", "avi", "mov"]
)

if yuklenen_dosya is not None:
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
        st.subheader("Canli Analiz Izleme")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            baslat = st.button("Analizi Baslat", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("Durdur", use_container_width=True):
                st.session_state["dur_analiz"] = True

    if baslat:
        st.session_state["dur_analiz"] = False

        with col2:
            kare_alani       = st.empty()
            ilerleme_bar     = st.progress(0)
            m1, m2           = st.columns(2)
            canli_skor_metni = m1.empty()
            fps_metni        = m2.empty()

        sonuc = process_video(
            tfile_name=tfile.name,
            model=model,
            mtcnn=mtcnn,
            donusum=donusum,
            fake_idx=fake_idx,
            esik_degeri=esik_degeri,
            frame_atlama=frame_atlama,
            min_yuz_boyutu=min_yuz_boyutu,
            temel_mtcnn_guven=temel_mtcnn_guven,
            ui_kare_alani=kare_alani,
            ui_ilerleme_bar=ilerleme_bar,
            ui_canli_skor_metni=canli_skor_metni,
            ui_fps_metni=fps_metni
        )

        if sonuc.get("hata"):
            st.error(f"Bir hata olustu: {sonuc['hata']}")
        else:
            st.divider()
            st.success("Analiz islemi tamamlandi.")

            st.subheader("Final Video Kaydi")
            if sonuc["islenmis_video_yolu"]:
                with open(sonuc["islenmis_video_yolu"], "rb") as f:
                    st.video(f.read())

            st.subheader("Akademik Analiz Raporu")

            tum_skorlar = sonuc["tum_skorlar"]
            gercek_tespitler = sonuc["gercek_tespitler"]
            tum_kareler = sonuc["tum_kareler"]
            ai_islem_suresi = sonuc["ai_islem_suresi"]
            mtcnn_hata_sayaci = sonuc["mtcnn_hata_sayaci"]
            sistem_fps = sonuc["sistem_fps"]

            if len(tum_skorlar) > 0:
                veri_kaynagi       = gercek_tespitler if len(gercek_tespitler) > 0 else tum_skorlar
                mean_skor          = float(np.mean(veri_kaynagi))
                percentile_85_skor = float(np.percentile(veri_kaynagi, 85))

                final_skor  = (0.4 * percentile_85_skor) + (0.6 * mean_skor)
                max_skor    = float(np.max(veri_kaynagi))
                tutarsizlik = temporal_tutarsizlik_skoru(tum_skorlar)

                gercek_ai_fps = (
                    len(tum_skorlar) / ai_islem_suresi
                    if ai_islem_suresi > 0 else 0.0
                )

                if final_skor > (esik_degeri * 100):
                    st.markdown(
                        f'<div class="alert-fake">DIKKAT: Deepfake manipulasyonu tespit edildi &nbsp;·&nbsp; '
                        f'Hibrit P85 Skoru: <b>%{final_skor:.1f}</b></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="alert-real">GUVENLI: Belirgin bir manipulasyon tespit edilmedi &nbsp;·&nbsp; '
                        f'Hibrit P85 Skoru: <b>%{final_skor:.1f}</b></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Genel Skor (Hibrit P85)", f"%{final_skor:.1f}")
                c2.metric("85. Persentil Skoru", f"%{percentile_85_skor:.1f}")
                c3.metric("Ortalama Skor", f"%{mean_skor:.1f}")
                c4.metric("Maksimum Anlik Skor", f"%{max_skor:.1f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Temporal Tutarsizlik (STD)", f"{tutarsizlik:.2f}")
                c6.metric("Yuz Tespit Edilen Kare", f"{len(gercek_tespitler)}")
                c7.metric("Saf AI Cikarim Hizi", f"{gercek_ai_fps:.1f} FPS")
                c8.metric("Toplam Sistem Hizi", f"{sistem_fps:.1f} FPS")

                if tutarsizlik > 20:
                    st.markdown(
                        f'<div class="alert-info"><b>Yuksek temporal tutarsizlik (STD={tutarsizlik:.1f})</b> — '
                        f'Video boyunca skor buyuk dalgalanmalar gosteriyor. '
                        f'Bu, anlik veya kesintili bir deepfake manipulasyonuna isaret edebilir.</div>',
                        unsafe_allow_html=True
                    )
                elif tutarsizlik < 5 and mean_skor > esik_degeri * 100:
                    st.markdown(
                        f'<div class="alert-info"><b>Dusuk temporal tutarsizlik (STD={tutarsizlik:.1f}) + yuksek ortalama</b> — '
                        f'Tutarli ve surekli bir deepfake manipulasyonu profili.</div>',
                        unsafe_allow_html=True
                    )

                if mtcnn_hata_sayaci > 0:
                    st.warning(f"MTCNN, {mtcnn_hata_sayaci} karede hata uretti. Bu kareler atlandi.")

                if len(tum_skorlar) > 1:
                    plot_timeline(tum_kareler, tum_skorlar, esik_degeri, percentile_85_skor)

                st.markdown("---")
                rapor_verisi = {
                    "analiz_tarihi"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sonuc"                     : "DEEPFAKE" if final_skor > (esik_degeri * 100) else "GERCEK",
                    "hibrit_p85_skoru_pct"      : round(final_skor, 2),
                    "p85_skoru_pct"             : round(percentile_85_skor, 2),
                    "ortalama_skor_pct"         : round(mean_skor, 2),
                    "max_anlik_skor_pct"        : round(max_skor, 2),
                    "temporal_tutarsizlik_std"  : round(tutarsizlik, 2),
                    "esik_degeri_pct"           : round(esik_degeri * 100, 1),
                    "analiz_edilen_kare_sayisi" : len(tum_skorlar),
                    "yuz_tespit_edilen_kare"    : len(gercek_tespitler),
                    "frame_skip"                : frame_atlama,
                    "mtcnn_hata_sayisi"         : mtcnn_hata_sayaci,
                    "model_mimari"              : "xception41",
                    "model_val_acc_pct"         : round(val_acc * 100, 2),
                    "egitim_epoch"              : egitim_epoch,
                    "temperature"               : TEMPERATURE,
                    "cihaz"                     : str(DEVICE),
                }
                st.download_button(
                    label="Analiz Raporunu Indir (JSON)",
                    data=json.dumps(rapor_verisi, ensure_ascii=False, indent=2),
                    file_name="deepfake_analiz_raporu.json",
                    mime="application/json",
                    use_container_width=True
                )