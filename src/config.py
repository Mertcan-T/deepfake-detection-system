import os
import torch

# ─────────────────────────────────────────────
#  DONANIM VE MİMARİ AYARLARI
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    USE_AMP = True
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False

# ─────────────────────────────────────────────
#  GİZLİ SİSTEM SABİTLERİ
# ─────────────────────────────────────────────
MODEL_YOLU = os.path.join(os.path.dirname(__file__), "deepfake_model.pth")

# T=1.5 ile softmax dağılımı gerçekten yumuşatılır; model aşırı özgüvenli tahminleri törpüler.
TEMPERATURE = 1.5

KAYIP_LIMITI           = 15   # Yüz art arda bu kadar kare kaybolursa skor sıfırlanır
MAX_BATCH_FACE         = 6    # VRAM taşmasını önlemek için maksimum yüz/kare
UI_GUNCELLEME_FREKANSI = 2    # Tarayıcı kilitlenmesini önleyen çizim seyreltici
