# Videolarda Deepfake Manipülasyonlarının Derin Öğrenme ile Tespiti

Bu proje, Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü **Python Programlamaya Giriş** dersi final projesi kapsamında geliştirilmiştir.

Sistem, üretken yapay zeka teknolojileri kullanılarak oluşturulan sahte videoları (**Deepfake**) tespit etmeyi amaçlayan uçtan uca çalışan bir derin öğrenme tabanlı analiz platformudur.

---

## Proje Özeti

Deepfake teknolojileri son yıllarda büyük gelişim göstermiş ve gerçek kişiler adına sahte video içerikleri üretmek mümkün hale gelmiştir. Bu durum bilgi güvenliği, medya doğrulama ve dijital kimlik güvenliği açısından önemli riskler oluşturmaktadır.

Bu proje kapsamında geliştirilen sistem:

* Videolardaki yüzleri otomatik olarak tespit eder.
* Derin öğrenme modeli ile analiz gerçekleştirir.
* Deepfake olasılığını hesaplar.
* Sonuçları grafikler ve raporlar ile kullanıcıya sunar.
* İşlenmiş videoyu kullanıcıya geri döndürür.
* Analiz raporunu JSON formatında dışa aktarır.

---

## Sistem Mimarisi

Sistem, iki aşamalı bir derin öğrenme boru hattı (pipeline) ve tamamen modüler, temiz bir proje yapısı kullanmaktadır.

### Proje Yapısı (Modüler Mimari)

Ölçeklenebilirlik ve bakım kolaylığı açısından projenin ana mantığı 5 farklı modüle bölünmüştür:

* **`src/app.py`**: Arayüzü başlatan ve bileşenleri birleştiren ana yönetici dosya.
* **`src/config.py`**: Model yolu, cihaz (GPU/CPU) tanımları ve karar eşikleri gibi sistem sabitlerini içerir.
* **`src/models.py`**: Derin öğrenme modellerinin (Xception41 ve MTCNN) önbelleğe alınarak yüklenmesinden sorumludur.
* **`src/video_processor.py`**: MTCNN yüz tespiti ve Xception analiz döngüsünü barındıran çekirdek işlem motorudur.
* **`src/ui.py`**: CSS tasarımları, uyarı bannerları ve Plotly zaman çizelgesi grafiği gibi arayüz çizim fonksiyonlarını barındırır.
* **`src/utils.py`**: Renk atamaları ve standart sapma hesabı gibi yardımcı fonksiyonları içerir.

---

### 1. Yüz Tespiti ve Hizalama (MTCNN)

Videodaki yüzler dinamik olarak tespit edilir.

Özellikler:

* Çoklu yüz desteği
* Dinamik güven eşikleri (merkez bölge için +0.10 ek hassasiyet)
* Profil ve geniş açı yüz desteği
* Yüz çevresine %10 güvenlik marjı (Domain Shift Koruması)

Kullanılan model:

* MTCNN (Multi-task Cascaded Convolutional Networks)

---

### 2. Deepfake Sınıflandırma (Xception41)

Tespit edilen yüzler:

* 299×299 boyutuna yeniden ölçeklendirilir.
* Normalize edilir.
* Xception41 modeline gönderilir.

Model;

* Piksel seviyesindeki bozulmaları,
* Sıkıştırma artefaktlarını,
* Yapay üretim izlerini,

analiz ederek görüntünün gerçek veya sahte olma olasılığını hesaplar.

Softmax çıktısı Temperature Scaling (T=1.5) ile yumuşatılarak modelin aşırı özgüvenli kararları törpülenir.

---

### 3. Hibrit Karar Mekanizması

Sistemde yalnızca ortalama skor kullanılmamaktadır.

Daha kararlı sonuçlar elde etmek amacıyla iki istatistik birlikte değerlendirilmektedir:

* **Ortalama Skor (Mean):** Videonun genel deepfake eğilimini ölçer.
* **85\. Persentil (P85):** Kısa süreli manipülasyon anlarını yakalamak için kullanılır.

**Önemli Uygulama Detayı:** P85 ve ortalama değerleri yalnızca yüzün gerçekten tespit edildiği kareler üzerinden hesaplanır. Forward-fill ile doldurulan (yüzün geçici olarak kaybolduğu) kareler bu hesaba dahil edilmez; böylece istatistiksel kirlilik önlenir.

Final karar skoru:

```text
Final Skor = (0.4 × P85) + (0.6 × Ortalama)
```

Bu yaklaşım kısa süreli manipülasyonların gözden kaçmasını önlemeye yardımcı olmaktadır.

---

### 4. Temporal Tutarsızlık Analizi

Sistem, video boyunca deepfake skorlarının standart sapmasını (STD) hesaplar:

| STD Değeri | Yorum |
| ---------- | ----- |
| STD > 20   | Anlık / kesintili deepfake — belirli karelerde yüksek, diğerlerinde düşük |
| STD < 5 + yüksek ortalama | Tutarlı, tam video deepfake manipülasyonu |
| STD < 5 + düşük ortalama  | Tutarlı gerçek video |

---

## Kullanılan Teknolojiler

| Teknoloji       | Amaç                                     |
| --------------- | ---------------------------------------- |
| Python          | Uygulamanın geliştirme dili              |
| PyTorch         | Derin öğrenme modeli eğitimi ve çıkarımı |
| TorchVision     | Görüntü dönüşümleri ve augmentation      |
| TIMM            | Xception41 model mimarisi                |
| Facenet-PyTorch | MTCNN yüz tespiti                        |
| OpenCV          | Video işleme                             |
| Streamlit       | Web arayüzü                              |
| NumPy           | Matematiksel işlemler                    |
| SciPy           | Savitzky-Golay filtreleme                |
| Plotly          | Grafik oluşturma                         |
| CUDA            | GPU hızlandırma                          |
| AMP (FP16)      | Mixed Precision hızlandırma              |

---

## Sistem Özellikleri

* Çoklu yüz analizi
* GPU hızlandırma (CUDA)
* Automatic Mixed Precision (FP16)
* Video bazlı deepfake analizi
* Canlı skor takibi
* İşlenmiş video çıktısı
* Savitzky-Golay sinyal filtreleme
* Hibrit P85 karar mekanizması (forward-fill'den bağımsız hesaplama)
* Temporal tutarsızlık (STD) analizi
* VRAM koruma sistemi
* Tarayıcı çökmesini önleyen UI optimizasyonları
* Analizi durdurma butonu
* JSON formatında rapor dışa aktarma
* Temperature Scaling (T=1.5) ile özgüven kalibrasyonu

---

## Veri Seti

Modelin eğitilmesinde Kaggle üzerinde yayınlanan aşağıdaki veri seti kullanılmıştır.

**Deepfake and Real Images Dataset**

Kaynak:

https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Veri seti:

* Gerçek yüz görüntüleri
* Deepfake görüntüler
* Manipüle edilmiş yüz örnekleri

içermektedir.

---

## Eğitim Stratejisi

### Model Mimarisi

Xception41, FaceForensics++ benchmark'ında deepfake tespiti için en yüksek başarıyı gösteren mimaridir. Transfer learning yaklaşımıyla ImageNet ön-eğitimli ağırlıklardan başlanarak fine-tune edilmiştir.

### Veri Ön İşleme ve Augmentation

Eğitim sırasında aşağıdaki augmentation teknikleri uygulanmıştır:

| Teknik | Amaç |
| ------ | ---- |
| RandomCrop (320→299) | Pozisyon çeşitliliği |
| RandomHorizontalFlip | Simetri bağımsızlığı |
| ColorJitter | Renk/ışık çeşitliliği |
| GaussianBlur | Video sıkıştırma ve motion blur simülasyonu |
| RandomGrayscale | Renk bağımsız özellik öğrenimi |
| RandomErasing | Kısmi yüz kapanmalarına (occlusion) dayanıklılık |

### Kayıp Fonksiyonu

`CrossEntropyLoss(label_smoothing=0.1)` kullanılmıştır. Label Smoothing, modelin "hard" 0/1 etiketler yerine kalibre edilmiş olasılıklar üretmesini sağlar ve implicit regularization olarak çalışır.

### Öğrenme Oranı Planlaması

`CosineAnnealingWarmRestarts (T₀=5)` kullanılmıştır. Her 5 epoch'ta öğrenme oranı yeniden başlatılarak lokal minimumlardan çıkılması hedeflenmiştir.

---

## Performans Sonuçları

### v1 Modeli (5 Epoch, Temel Augmentation)

| Metrik              | Sonuç                      |
| ------------------- | -------------------------- |
| Eğitim Accuracy     | %99.21                     |
| Validation Accuracy | %98.76                     |
| En İyi Epoch        | 5                          |
| Mimari              | Xception41                 |
| Framework           | PyTorch                    |
| Donanım             | NVIDIA RTX 4050 Laptop GPU |

### v2 Modeli (15 Epoch, Güçlendirilmiş Strateji)

| Metrik              | Sonuç                          |
| ------------------- | ------------------------------ |
| Epoch Sayısı        | 15                             |
| Kayıp Fonksiyonu    | CrossEntropy + Label Smoothing |
| Scheduler           | CosineAnnealingWarmRestarts    |
| Mimari              | Xception41                     |
| Framework           | PyTorch                        |
| Donanım             | NVIDIA RTX 4050 Laptop GPU     |

> Not: v2 model eğitim sonuçları tamamlandıktan sonra buraya eklenecektir.

---

## Kurulum

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/Mertcan-T/deepfake-detection-system.git
cd deepfake-detection-system
```

### 2. Gerekli Kütüphaneleri Kurun

```bash
pip install -r requirements.txt
```

### 3. Model Dosyasını Yerleştirin

`deepfake_model.pth` dosyası GitHub dosya boyutu sınırları nedeniyle repoya eklenmemiştir.

Model dosyasını indirerek aşağıdaki konuma yerleştirin:

```text
src/deepfake_model.pth
```

**İsteğe bağlı:** Modeli kendiniz eğitmek için:

```bash
python src/train.py           # Yeni eğitim (15 epoch)
python src/train.py --resume  # Kalınan epoch'tan devam
```

### 4. Uygulamayı Başlatın

```bash
streamlit run src/app.py
```

---

## Çalışma Akışı

1. Kullanıcı videoyu sisteme yükler.
2. Video karelere ayrılır (Frame Skip ile hız optimizasyonu).
3. MTCNN ile yüz tespiti yapılır (dinamik güven eşikleri + %10 padding).
4. Tespit edilen yüzler Xception41 modeline gönderilir (Batch Inference + AMP FP16).
5. Softmax çıktısı Temperature Scaling (T=1.5) ile kalibre edilir.
6. Deepfake olasılıkları hesaplanır; Forward-Fill ile geçici yüz kayıpları giderilir.
7. Yalnızca gerçek yüz tespiti olan kareler üzerinden Hibrit P85 karar skoru hesaplanır.
8. Temporal tutarsızlık (STD) analizi yapılır.
9. Sonuçlar grafik, işlenmiş video ve JSON rapor olarak kullanıcıya sunulur.

---

## Proje Ekibi

**Mertcan TAŞKIRAN**
032290113

**Füsun GÜN**
032490014

**Büşra DERELİ**
032490047

---

## Lisans

Bu proje Bursa Uludağ Üniversitesi Bilgisayar Mühendisliği Bölümü kapsamında eğitim amaçlı geliştirilmiştir.

Ticari kullanım amacı taşımamaktadır.
