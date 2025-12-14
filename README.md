# Demo Klasifikasi Cats vs Dogs - Model Pipeline

Aplikasi Streamlit untuk mengklasifikasi audio kucing vs anjing menggunakan machine learning pipeline (SelectKBest + RandomForest).

## Struktur Proyek

```
d:\Project Akhir PSD\
├── app.py                              # Aplikasi Streamlit utama
├── requirements.txt                    # Dependensi Python
├── model_pipeline.joblib               # Model yang sudah dilatih (SelectKBest + RF)
├── data_processed.npz                  # Data contoh untuk demo (X_test, y_test, labels)
├── preprocessing_pipeline.joblib       # Pipeline preprocessing (untuk referensi)
├── label_encoder.joblib                # Label encoder (untuk referensi)
├── .streamlit/config.toml              # Konfigurasi Streamlit
├── Dockerfile                          # Build Docker image (opsional)
├── dataset/
│   └── CatsDogs_TRAIN.arff             # Dataset training asli
├── figures/
│   └── confusion_matrix_selectk_best.png  # Gambar hasil training
├── Data Understanding.ipynb            # EDA notebook (BAB II)
├── preprocessing.ipynb                 # Preprocessing notebook (BAB III)
├── modeling.ipynb                      # Modeling notebook (BAB IV)
└── cv_results*.txt, test_metrics*.txt  # Hasil metrics & CV
```

## Setup Lokal (Windows)

### 1. Persiapan Awal
```cmd
cd "D:\Project Akhir PSD"
```

### 2. Buat Virtual Environment (Rekomendasi)
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Instal Dependensi
```cmd
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```cmd
streamlit run app.py
```

Aplikasi akan membuka di browser: `http://localhost:8501`

### 5. Fitur Aplikasi
- **Prediksi dari Test Set**: Pilih contoh dari 33 sampel test menggunakan slider, lihat prediksi model dan label sebenarnya.
- **Upload Audio (Eksperimental)**: Upload file `.wav` atau `.mp3` untuk ekstraksi MFCC dan prediksi. (Catatan: ekstraksi MFCC adalah pendekatan eksperimental; model dilatih pada fitur berbeda, hasil mungkin tidak akurat.)
- **Confusion Matrix**: Tampilkan matriks kebingungan dari evaluasi test set (SelectKBest pipeline).

## Deploy ke Streamlit Cloud

### 1. Siapkan Repository GitHub
```bash
# Clone atau inisialisasi repo lokal
git init
git add .
git commit -m "Initial commit: Cats vs Dogs classifier demo"
git branch -M main
git remote add origin https://github.com/username/project-akhir-psd.git
git push -u origin main
```

### 2. Daftar di Streamlit Cloud
1. Buka https://streamlit.io/cloud
2. Sign up dengan akun GitHub
3. Klik "New app"
4. Pilih repository & branch (`main`), folder (`.`), file (`app.py`)
5. Deploy

### 3. Konfigurasi Streamlit Cloud
- Streamlit Cloud akan membaca `.streamlit/config.toml` secara otomatis.
- Jika diperlukan, edit di dashboard Streamlit Cloud → Settings → Advanced settings.

**URL hasil deploy**: `https://username-projectname.streamlit.app/`

## Deploy Menggunakan Docker (Opsional)

### 1. Build Image
```bash
docker build -t cats-dogs-classifier:latest .
```

### 2. Jalankan Container Lokal
```bash
docker run -p 8501:8501 cats-dogs-classifier:latest
```

### 3. Deploy ke Cloud (mis. Render, Google Cloud, AWS)
- Render: Push image ke Docker Hub, hubungkan dengan Render, deploy sebagai Web Service
- Google Cloud Run: `gcloud run deploy --source .`
- AWS ECS: Tag image, push ke ECR, deploy ke ECS

## Performa Model

- **CV F1-macro (nested CV)**: 0.525 ± 0.058
- **Test Accuracy**: 0.636
- **Test F1-macro**: 0.636
- **Pipeline**: VarianceThreshold → StandardScaler → SelectKBest(k=200) → RandomForestClassifier(n_estimators=100)

Catatan: Dataset kecil (~130 train, 33 test) dan dimensi fitur tinggi (~14,773 awal) menyebabkan performa moderat. Untuk improvement: augmentasi data, ekstraksi fitur lebih baik (MFCC/statistik), atau kumpulkan lebih banyak sampel.

## Troubleshooting

### Error: "No module named 'numpy._core'"
- **Penyebab**: NumPy tidak terpasang atau korup
- **Solusi**:
  ```cmd
  pip install --force-reinstall numpy
  ```

### Error: "Model tidak ditemukan"
- **Penyebab**: File `model_pipeline.joblib` tidak ada
- **Solusi**: Jalankan notebook `modeling.ipynb` hingga selesai untuk menghasilkan model

### Prediksi Audio Gagal
- **Penyebab**: `librosa` tidak terpasang, atau dimensi fitur tidak cocok
- **Solusi**:
  - Pasang `librosa`: `pip install librosa`
  - Catatan: ekstraksi MFCC sederhana (80 fitur) mungkin tidak kompatibel dengan model (yang dilatih pada ~200+ fitur setelah SelectKBest)

## Informasi Tambahan

- **Dataset**: CatsDogs time-series audio (ARFF format)
- **Referensi**: Piczak (2015), Salamon et al. (2014)
- **Teknologi**: Python, scikit-learn, Streamlit, joblib
- **Maintenance**: Model static; untuk update model, jalankan notebook modeling dan replace `model_pipeline.joblib`

## Kontak & Dokumentasi

Untuk detil lebih lanjut tentang preprocessing dan modeling, lihat:
- `Data Understanding.ipynb` → BAB II (EDA)
- `preprocessing.ipynb` → BAB III (Preprocessing)
- `modeling.ipynb` → BAB IV (Modeling & Evaluation)
