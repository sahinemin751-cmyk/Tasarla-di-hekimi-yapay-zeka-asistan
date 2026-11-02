# Diş hekimi yapay zeka asistanı (prototip)

Loup kamerası ve radyografiyle hekime "ikinci göz" olacak asistanın ilk prototipi. Bu sürüm tek bir şeyi yapar: ağız içi kamera görüntüsünde **çürük şüphesini** işaretler.

> **Araştırma prototipidir, klinik karar için değildir.** Tıbbi cihaz sertifikası yoktur.

Ürün ve teknik tasarım: [tasarım dokümanı](https://claude.ai/artifact/1FmfDcJ29Ph9ifLrNWq5rK)

## Ne var

| Parça | Dosya | Ne yapar |
| --- | --- | --- |
| Veri hazırlama | `scripts/prepare_caries_dataset.py` | Zenodo çürük veri setini YOLO eğitim formatına getirir |
| Eğitim | `scripts/train_caries.py` | YOLO11 çürük modelini eğitir, test setinde duyarlılık ve mAP verir |
| Kamera demosu | `scripts/camera_demo.py` | USB kameradan canlı görüntü alır, net kareleri modele verir, birkaç karede teyit edilen bulguları çizer |
| Kalite kapısı | `dental_asistan/quality.py` | Bulanık, karanlık ve parlamalı kareleri eler |
| Zamansal oylama | `dental_asistan/temporal.py` | Tek karede görülen tespiti yok sayar, tekrar edenleri teyit eder |

## Ekran kartın yoksa: Colab ile eğit

Modeli Google Colab'ın ücretsiz GPU'sunda eğitmek en kolay yol. [Defteri Colab'da aç](https://colab.research.google.com/github/sahinemin751-cmyk/Tasarla-di-hekimi-yapay-zeka-asistan/blob/main/notebooks/colab_egitim.ipynb), T4 GPU seç ve "Tümünü çalıştır" de. Veri indirme, hazırlama, eğitim ve test kendiliğinden yapılır; sonunda `best.pt` bilgisayarına iner. Kamera demosu ekran kartı olmadan da çalışır.

## Windows'ta kurulum

1. [Python 3.11 veya 3.12](https://www.python.org/downloads/windows/) kur (kurulumda "Add python.exe to PATH" işaretli olsun).
2. NVIDIA ekran kartın varsa güncel sürücüyü kur. PyTorch'un CUDA sürümü için [pytorch.org](https://pytorch.org/get-started/locally/) adresindeki komutu kullan.
3. Komut isteminde:

```bat
git clone https://github.com/sahinemin751-cmyk/Tasarla-di-hekimi-yapay-zeka-asistan.git
cd Tasarla-di-hekimi-yapay-zeka-asistan
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```

## Kullanım

**1. Veriyi indir ve hazırla.** [Zenodo 14827784](https://zenodo.org/records/14827784) sayfasından `Benchmarking Dataset.zip` dosyasını indir (1,6 GB; yazarların train/valid/test ayrımlı sürümü), sonra:

```bat
python scripts\prepare_caries_dataset.py --source "C:\Users\<sen>\Downloads\Benchmarking Dataset.zip" --out data\caries
```

Betik arşivdeki YOLO etiketlerini kendisi bulur. Klasör adlarında train/val/test varsa o ayrımı korur, yoksa %80/%10/%10 böler.

**2. Modeli eğit.**

```bat
python scripts\train_caries.py --data data\caries\data.yaml --epochs 100
```

Ekran kartı yoksa eğitim saatler sürer. O durumda aynı komutları Google Colab'da (ücretsiz GPU) çalıştırabilirsin. Çıktı: `runs\caries\weights\best.pt` ve test setinde precision, recall ve mAP değerleri.

**3. Kamerayla dene.**

```bat
python scripts\camera_demo.py --weights runs\caries\weights\best.pt --camera 0
```

`s` tuşu kareyi ve tespitleri `captures\` klasörüne kaydeder, `q` çıkar. Kamera bulanık görünüyorsa `--min-sharpness` değerini düşür.

## Veri ve lisans

- Çürük verisi: Ahmed ve ark., *Annotated intraoral image dataset for dental caries detection*, Zenodo, 2025, [doi:10.5281/zenodo.14827784](https://doi.org/10.5281/zenodo.14827784), CC BY 4.0. Bu veriyle eğitilen modeller kullanılırken kaynak belirtilmelidir.
- Premalign lezyon verisi (Piyarathne ve ark., Oral Oncology 2024) CC BY-NC-ND lisanslıdır. Ticari üründe kullanılamaz, bu yüzden bu depoya henüz eklenmedi.
- Hasta görüntüleri, eğitim çıktıları ve model ağırlıkları `.gitignore` ile depodan dışarıda tutulur. KVKK gereği hasta verisi asla bu depoya yüklenmemelidir.

## Sınırlar

- Veri seti intraoral kamera fotoğraflarından oluşur, loup kamerası görüntüsü değildir. Loup kamerasında performans ayrıca ölçülmelidir.
- Veri Pakistan'daki bir hastaneden gelir. Türk hasta popülasyonunda doğrulanmamıştır.
- Aproksimal (diş arası) çürükler kamerada çoğunlukla görünmez. Onlar radyografi modülünün işidir (sonraki adım).
