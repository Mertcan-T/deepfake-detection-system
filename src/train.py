"""
============================================================
  Deepfake Tespit Sistemi — Model Eğitim Scripti (train.py)
============================================================
DEĞİŞİKLİK GEÇMİŞİ:

  v1 → v2: InceptionResnetV1 → xception41, CUDA, Transfer Learning,
           EarlyStopping, Checkpoint formatı, PowerShell buffer.
  v2 → v3: RESUME özelliği, optimizer/scheduler state kaydı.
  v3 → v4: AMP (FP16) eklendi, GradScaler, cudnn.benchmark.
  v4 → v5 (FİNAL): GradScaler import düzeltildi, optimizer device uyuşmazlığı çözüldü.
  v5 → v6 (GELİŞTİRİLMİŞ):
    - EPOCHS: 5 → 15 (daha iyi genelleme)
    - Label Smoothing: CrossEntropyLoss(label_smoothing=0.1) eklendi.
      Model aşırı özgüvenli "hard" tahminler yerine daha kalibre edilmiş
      olasılıklar üretiyor. Overfitting'e karşı implicit regularization.
    - Güçlendirilmiş Augmentation:
        * GaussianBlur → Video sıkıştırma artefaktlarını ve motion blur'u simüle eder.
        * RandomErasing → Kısmi yüz kapanmalarına (occlusion) karşı dayanıklılık.
        * RandomGrayscale → Renk bağımsız özellikler öğrenmesini teşvik eder.
        * Daha güçlü ColorJitter parametreleri.
    - Scheduler: ReduceLROnPlateau → CosineAnnealingWarmRestarts
      Periyodik LR artışları lokal minimumlardan çıkmayı sağlar.
      Resume ile uyumludur.

KULLANIM:
  python train.py           # Normal eğitim
  python train.py --resume  # Kalınan epoch'tan devam
============================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────
VERI_DIZINI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "deepfake_data", "Dataset"))
MODEL_KAYIT = os.path.join(os.path.dirname(__file__), "deepfake_model.pth")

# v6: Epoch sayısı artırıldı — transfer learning ile 15 epoch yeterli genelleme sağlar
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-4
IMG_SIZE    = 299
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = torch.cuda.is_available()
RESUME      = "--resume" in sys.argv

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"\n{'='*55}")
print(f"  Cihaz : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
print(f"  AMP   : {'Aktif (FP16)' if USE_AMP else 'Pasif (FP32)'}")
print(f"  Mod   : {'RESUME (kalınan yerden devam)' if RESUME else 'YENİ EĞİTİM'}")
print(f"  Epoch : {EPOCHS}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────
#  VERİ — v6 Güçlendirilmiş Augmentation
# ─────────────────────────────────────────────
egitim_donusum = transforms.Compose([
    # Hafif büyütme sonrası merkez kırpma → pozisyon çeşitliliği
    transforms.Resize((320, 320)),
    transforms.RandomCrop(IMG_SIZE),

    transforms.RandomHorizontalFlip(),

    # v6: Daha güçlü renk dönüşümleri
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),

    # v6 YENİ: Motion blur ve video sıkıştırma artefaktlarını simüle eder.
    # Eğitim veri seti statik görüntülerden oluştuğundan bu augmentation
    # video karelerine domain shift'i azaltır.
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

    # v6 YENİ: Renk bağımsız özellikler öğrenmesini teşvik eder (%5 olasılıkla)
    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

    # v6 YENİ: Kısmi yüz kapanmalarına (occlusion) karşı dayanıklılık.
    # Gerçek video senaryolarında eller, saç veya nesneler yüzü kısmen kapatabilir.
    transforms.RandomErasing(p=0.10, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
])

deger_donusum = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

egitim_veri    = datasets.ImageFolder(os.path.join(VERI_DIZINI, "Train"),      transform=egitim_donusum)
dogrulama_veri = datasets.ImageFolder(os.path.join(VERI_DIZINI, "Validation"), transform=deger_donusum)
egitim_yukleme    = DataLoader(egitim_veri,    batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
dogrulama_yukleme = DataLoader(dogrulama_veri, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Sınıf eşlemesi : {egitim_veri.class_to_idx}")
print(f"Eğitim         : {len(egitim_veri):,} görüntü")
print(f"Doğrulama      : {len(dogrulama_veri):,} görüntü\n")

# ─────────────────────────────────────────────
#  MODEL, OPTIMIZER, SCALER
# ─────────────────────────────────────────────
model = timm.create_model("xception41", pretrained=not RESUME, num_classes=2)

# v6 YENİ: Label Smoothing
# Modelin "hard" 0/1 tahminler yerine kalibre edilmiş olasılıklar üretmesini sağlar.
# epsilon=0.1 → hedef etiketler [0.05, 0.95] aralığına yumuşatılır.
# Implicit regularization olarak çalışır; overfitting'i azaltır.
kayip_fonk = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# v6 YENİ: CosineAnnealingWarmRestarts
# ReduceLROnPlateau'ya göre avantajı: periyodik LR artışları lokal minimumlardan
# çıkmayı sağlar. T_0=5 → her 5 epoch'ta bir LR sıfırlanır.
# Resume ile uyumludur (scheduler state checkpoint'e kaydedilir).
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ─────────────────────────────────────────────
#  RESUME KONTROLÜ
# ─────────────────────────────────────────────
baslangic_epoch = 1
en_iyi_acc      = 0.0

# MİMARİ DÜZELTME: Model her zaman checkpoint yüklenmeden ÖNCE GPU'ya taşınır!
model = model.to(DEVICE)
print(f"Model {DEVICE} cihazına taşındı.")

if RESUME and os.path.exists(MODEL_KAYIT):
    print(f"Checkpoint yükleniyor: {MODEL_KAYIT}")

    checkpoint = torch.load(MODEL_KAYIT, map_location=DEVICE, weights_only=False)

    # 1. Model Ağırlıkları
    model.load_state_dict(checkpoint["model_state"])
    en_iyi_acc = checkpoint.get("val_acc", 0.0)

    # 2. Optimizer
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("  ✓ Optimizer state yüklendi")

    # 3. Scheduler ve Scaler
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("  ✓ Scheduler state yüklendi")

    if "scaler_state" in checkpoint and USE_AMP:
        scaler.load_state_dict(checkpoint["scaler_state"])
        print("  ✓ GradScaler state yüklendi")

    baslangic_epoch = checkpoint.get("son_epoch", checkpoint.get("epoch", 1)) + 1
    print(f"  ✓ Epoch {baslangic_epoch}'dan devam ediliyor")
    print(f"  ✓ Mevcut en iyi Val Acc: %{en_iyi_acc*100:.2f}\n")
elif RESUME:
    print("Uyarı: Checkpoint bulunamadı, sıfırdan başlanıyor.\n")

print(f"Eğitim Epoch {baslangic_epoch}/{EPOCHS}'dan başlıyor.\n")

# ─────────────────────────────────────────────
#  EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────
for epoch in range(baslangic_epoch, EPOCHS + 1):

    model.train()
    toplam_kayip = 0.0
    dogru        = 0
    t0           = time.time()

    for i, (goruntu, etiket) in enumerate(egitim_yukleme):
        goruntu, etiket = goruntu.to(DEVICE), etiket.to(DEVICE)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if USE_AMP else "cpu", enabled=USE_AMP):
            cikti = model(goruntu)
            kayip = kayip_fonk(cikti, etiket)

        scaler.scale(kayip).backward()
        scaler.step(optimizer)
        scaler.update()

        toplam_kayip += kayip.item()
        dogru        += (cikti.argmax(1) == etiket).sum().item()

        if (i + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {i+1}/{len(egitim_yukleme)} | "
                  f"Kayıp: {toplam_kayip/(i+1):.4f}", flush=True)

    egitim_acc = dogru / len(egitim_veri)

    # — Doğrulama —
    model.eval()
    val_dogru = 0
    with torch.no_grad():
        for goruntu, etiket in dogrulama_yukleme:
            goruntu, etiket = goruntu.to(DEVICE), etiket.to(DEVICE)
            with torch.autocast(device_type="cuda" if USE_AMP else "cpu", enabled=USE_AMP):
                cikti = model(goruntu)
            val_dogru += (cikti.argmax(1) == etiket).sum().item()

    val_acc   = val_dogru / len(dogrulama_veri)
    sure      = time.time() - t0
    mevcut_lr = optimizer.param_groups[0]['lr']

    print(f"\nEpoch {epoch}/{EPOCHS} | "
          f"Eğitim Acc: %{egitim_acc*100:.2f} | "
          f"Val Acc: %{val_acc*100:.2f} | "
          f"LR: {mevcut_lr:.2e} | "
          f"Süre: {sure/60:.1f}dk", flush=True)

    # CosineAnnealingWarmRestarts her epoch sonunda güncellenir
    scheduler.step(epoch + (1 / len(egitim_yukleme)))

    if val_acc > en_iyi_acc:
        en_iyi_acc = val_acc
        os.makedirs(os.path.dirname(MODEL_KAYIT), exist_ok=True)
        torch.save({
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "mimari":          "xception41",
            "num_classes":     2,
            "class_to_idx":    egitim_veri.class_to_idx,
            "val_acc":         val_acc,
            "epoch":           epoch,
            "son_epoch":       epoch,
        }, MODEL_KAYIT)
        print(f"  ✓ En iyi model kaydedildi → Val Acc: %{val_acc*100:.2f}", flush=True)

print(f"\n{'='*55}")
print(f"  Eğitim tamamlandı!")
print(f"  En iyi Val Accuracy : %{en_iyi_acc*100:.2f}")
print(f"  Model               : {MODEL_KAYIT}")
print(f"{'='*55}")