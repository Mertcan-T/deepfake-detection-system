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

---

## Sistem Mimarisi

Sistem iki aşamalı bir derin öğrenme boru hattı (pipeline) kullanmaktadır.

### 1. Yüz Tespiti ve Hizalama (MTCNN)

Videodaki yüzler dinamik olarak tespit edilir.

Özellikler:

* Çoklu yüz desteği
* Dinamik güven eşikleri
* Profil ve geniş açı yüz desteği
* Yüz çevresine güvenlik marjı (Padding) ekleme

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

---

### 3. Hibrit Karar Mekanizması

Sistemde yalnızca ortalama skor kullanılmamaktadır.

Daha kararlı sonuçlar elde etmek amacıyla:

* Ortalama Skor (Mean)
* 90. Persentil (P90)

birlikte değerlendirilmektedir.

Final karar skoru:

```text
Final Skor = (0.7 × P90) + (0.3 × Ortalama)
```

Bu yaklaşım kısa süreli manipülasyonların gözden kaçmasını önlemeye yardımcı olmaktadır.

---

## Kullanılan Teknolojiler

| Teknoloji       | Amaç                                     |
| --------------- | ---------------------------------------- |
| Python          | Uygulamanın geliştirme dili              |
| PyTorch         | Derin öğrenme modeli eğitimi ve çıkarımı |
| TorchVision     | Görüntü dönüşümleri                      |
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
* Hibrit P90 karar mekanizması
* VRAM koruma sistemi
* Tarayıcı çökmesini önleyen UI optimizasyonları

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

## Performans Sonuçları

Model eğitimi sonucunda aşağıdaki performans değerleri elde edilmiştir.

| Metrik              | Sonuç                      |
| ------------------- | -------------------------- |
| Eğitim Accuracy     | %99.21                     |
| Validation Accuracy | %98.76                     |
| En İyi Epoch        | 5                          |
| Mimari              | Xception41                 |
| Framework           | PyTorch                    |
| Donanım             | NVIDIA RTX 4050 Laptop GPU |

Not: Sonuçlar kullanılan veri bölünmesine ve rastgele başlangıç değerlerine bağlı olarak küçük farklılıklar gösterebilir.


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

### 4. Uygulamayı Başlatın

```bash
streamlit run src/app.py
```

---

## Çalışma Akışı

1. Kullanıcı videoyu sisteme yükler.
2. Video karelere ayrılır.
3. MTCNN ile yüz tespiti yapılır.
4. Tespit edilen yüzler Xception41 modeline gönderilir.
5. Deepfake olasılıkları hesaplanır.
6. Hibrit P90 karar mekanizması uygulanır.
7. Sonuçlar grafik ve işlenmiş video olarak kullanıcıya sunulur.

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
