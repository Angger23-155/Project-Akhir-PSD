# Panduan Deploy ke Streamlit Cloud (untuk Dosen)

Langkah cepat untuk deploy aplikasi Cats vs Dogs classifier ke cloud (gratis, public URL, dapat diakses siapa saja).

## Persiapan (15 menit)

### 1. Siapkan Repository GitHub
Anda perlu:
- Akun GitHub (daftar gratis di https://github.com jika belum ada)
- Git installed di komputer (https://git-scm.com/download/win)

### 2. Inisialisasi Git Repository Lokal
Buka cmd di folder project:
```cmd
cd "D:\Project Akhir PSD"
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git add .
git commit -m "Initial commit: Cats vs Dogs classifier with Streamlit demo"
```

### 3. Buat Repository Baru di GitHub
1. Buka https://github.com/new
2. Nama repo: `Project-Akhir-PSD` (atau nama lain)
3. Klik "Create repository"
4. Salin perintah "…or push an existing repository from the command line"
5. Jalankan di cmd (sesuaikan URL):
```cmd
git branch -M main
git remote add origin https://github.com/username/Project-Akhir-PSD.git
git push -u origin main
```
Ganti `username` dengan username GitHub Anda.

### 4. Daftar & Buat Akun Streamlit Cloud
1. Buka https://streamlit.io/cloud
2. Klik "Sign up" → Pilih "Sign up with GitHub"
3. Autorize Streamlit
4. Setelah login, klik "New app" → "Create app"

### 5. Deploy Aplikasi
Di halaman "Create app", isi:
- **Repository**: Pilih `username/Project-Akhir-PSD` (repo yang baru dibuat)
- **Branch**: `main` (default)
- **Main file path**: `app.py`
- Klik "Deploy"

Streamlit akan membangun dan meluncurkan aplikasi. Tunggu hingga selesai (2-5 menit).

### 6. Akses Aplikasi
Setelah deploy selesai, Streamlit Cloud akan memberikan URL publik:
```
https://project-akhir-psd-username.streamlit.app/
```

URL ini dapat dibagikan ke dosen, penguji, atau publik. Aplikasi dapat diakses dari browser mana saja (Windows, macOS, Linux, smartphone).

## Troubleshooting Deployment

### Error: "File not found: model_pipeline.joblib"
- **Penyebab**: File model belum di-push ke GitHub
- **Solusi**:
  ```cmd
  git add model_pipeline.joblib data_processed.npz figures/
  git commit -m "Add model and data files"
  git push
  ```
  Tunggu Streamlit Cloud rebuild (biasanya otomatis dalam beberapa detik).

### Error: "ModuleNotFoundError: No module named 'X'"
- **Penyebab**: Dependensi belum di-list di `requirements.txt`
- **Solusi**:
  1. Edit `requirements.txt` lokal, tambahkan paket yang kurang
  2. Push ke GitHub:
     ```cmd
     git add requirements.txt
     git commit -m "Update requirements"
     git push
     ```
  3. Streamlit Cloud akan rebuild otomatis

### Aplikasi lambat atau timeout
- **Penyebab**: Model besar atau ekstraksi audio memakan waktu
- **Solusi**: 
  - Untuk Streamlit Cloud, processing MFCC mungkin slow; pertimbangkan disable atau caching
  - Edit `app.py`, tambahkan `@st.cache_data` di atas fungsi ekstraksi audio

### Ingin update model atau perubahan kode
1. Edit file lokal
2. Push ke GitHub:
   ```cmd
   git add .
   git commit -m "Update: deskripsi perubahan"
   git push
   ```
3. Streamlit Cloud akan rebuild otomatis dalam 1-2 menit

## URL Deploymen Contoh

Setelah sukses, URL Anda akan terlihat seperti:
```
https://project-akhir-psd-123abc.streamlit.app/
```

Bagikan URL ini ke:
- Dosen pembimbing
- Penguji tugas akhir
- Teman / stakeholder
- Lampirkan di laporan / presentasi

## Tips untuk Presentasi

1. **Test URL sebelum presentasi**: Buka URL di browser untuk memastikan berfungsi
2. **Screenshot URL**: Ambil screenshot halaman pertama aplikasi untuk dokumentasi
3. **Demo lokal backup**: Siapkan juga demo lokal (`streamlit run app.py`) sebagai backup jika cloud down
4. **Performa**: Jangan upload file audio besar (>10MB) di demo; gunakan contoh test set untuk demo yang stabil

## Biaya & Quota

- **Gratis**: Unlimited deployments, 1 GB RAM per aplikasi, 5 concurrent users
- **Bayar** (opsional): Jika perlu lebih banyak resources

Untuk dosen/akademik, free tier sudah cukup.

## Soal Keamanan & Privacy

- Model dan kode bersifat publik (di GitHub)
- File `.streamlit/secrets.toml` (jika ada API keys) harus di-add ke `.gitignore` (sudah ada)
- Data yang di-upload user tidak disimpan (demo hanya read-only)

## Next Steps

Setelah deploy:
1. ✅ Test aplikasi di cloud (buka URL)
2. ✅ Verifikasi fitur berfungsi
3. ✅ Bagikan URL ke dosen
4. (Opsional) Setup custom domain jika ingin (Streamlit Cloud support custom domain)

---

**Butuh bantuan lagi?** Chat dengan saya atau lihat dokumentasi Streamlit: https://docs.streamlit.io/deploy/streamlit-cloud
