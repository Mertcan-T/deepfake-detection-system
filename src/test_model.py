import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random

MODEL_YOLU = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\deepfake_model.pth"
TEST_DIZINI = r"C:\Users\M.T\Desktop\calismalarim\DeepFake Python\deepfake-detection-system\src\deepfake_data"

checkpoint = torch.load(MODEL_YOLU, map_location="cpu")
model = timm.create_model("xception41", pretrained=False, num_classes=2)
model.load_state_dict(checkpoint["model_state"])
model.eval()

donusum = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dogru = 0
toplam = 0

for cls in ["Real", "Fake"]:
    klasor = os.path.join(TEST_DIZINI, cls)
    dosyalar = random.sample(os.listdir(klasor), 10)
    print(f"\n--- {cls} Goruntuleri ---")
    for d in dosyalar:
        img = Image.open(os.path.join(klasor, d)).convert("RGB")
        tensor = donusum(img).unsqueeze(0)
        with torch.no_grad():
            cikti = model(tensor)
            olas = F.softmax(cikti, dim=1)
            fake_s = olas[0][0].item()
            real_s = olas[0][1].item()
        tahmin = "FAKE" if fake_s > 0.5 else "REAL"
        dogru_mu = (tahmin == cls.upper())
        dogru += int(dogru_mu)
        toplam += 1
        isaret = "✓" if dogru_mu else "✗"
        print(f"  {isaret} {d[:25]:25s} → Fake: %{fake_s*100:.1f} | Real: %{real_s*100:.1f} | {tahmin}")

print(f"\n{'='*50}")
print(f"  Test Dogrulugu: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")
print(f"{'='*50}")