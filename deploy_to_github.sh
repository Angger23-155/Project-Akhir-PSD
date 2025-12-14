#!/bin/bash
# Script otomatis untuk deploy ke Streamlit Cloud (Linux/Mac)
# Pastikan Anda sudah:
# 1. Install Git (sudah ada di kebanyakan Linux/Mac)
# 2. Buat akun GitHub (https://github.com)
# 3. Buat repository kosong di GitHub (https://github.com/new)

echo "============================================"
echo "Deploy Cats vs Dogs Classifier ke Streamlit Cloud"
echo "============================================"
echo ""

# Input dari user
read -p "Masukkan GitHub username Anda: " GITHUB_USERNAME
read -p "Masukkan nama repository GitHub (default: Project-Akhir-PSD): " REPO_NAME
REPO_NAME=${REPO_NAME:-Project-Akhir-PSD}
read -p "Masukkan nama Anda (untuk git config): " USER_NAME
read -p "Masukkan email Anda (untuk git config): " USER_EMAIL

echo ""
echo "Konfigurasi:"
echo "  GitHub Username: $GITHUB_USERNAME"
echo "  Repository Name: $REPO_NAME"
echo "  Git User: $USER_NAME <$USER_EMAIL>"
echo ""

# Cek apakah git terinstall
if ! command -v git &> /dev/null; then
    echo "ERROR: Git tidak terinstall"
    echo "Install Git dengan: apt-get install git (Linux) atau brew install git (Mac)"
    exit 1
fi

# Inisialisasi git lokal
echo "[1/6] Inisialisasi repository lokal..."
git init
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal menjalankan git init"
    exit 1
fi

# Konfigurasi user
echo "[2/6] Konfigurasi git user..."
git config user.name "$USER_NAME"
git config user.email "$USER_EMAIL"

# Add semua file
echo "[3/6] Menambahkan semua file..."
git add .
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal menjalankan git add"
    exit 1
fi

# Commit
echo "[4/6] Melakukan commit..."
git commit -m "Initial commit: Cats vs Dogs classifier dengan Streamlit demo"
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal melakukan commit"
    exit 1
fi

# Rename branch ke main
echo "[5/6] Rename branch ke 'main'..."
git branch -M main

# Add remote dan push
echo "[6/6] Push ke GitHub..."
echo ""
echo "   URL Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal add remote"
    exit 1
fi

git push -u origin main
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal push. Pastikan:"
    echo "   1. Repository sudah dibuat di GitHub"
    echo "   2. URL GitHub username dan nama repo benar"
    echo "   3. Anda sudah login GitHub di git (SSH key / token)"
    exit 1
fi

echo ""
echo "============================================"
echo "SUCCESS! Push ke GitHub selesai."
echo "============================================"
echo ""
echo "Langkah selanjutnya:"
echo "1. Buka https://streamlit.io/cloud"
echo "2. Sign up / Login dengan GitHub"
echo "3. Klik 'New app' -> 'Create app'"
echo "4. Pilih repository: $GITHUB_USERNAME/$REPO_NAME"
echo "5. Branch: main"
echo "6. Main file path: app.py"
echo "7. Klik 'Deploy'"
echo ""
echo "Tunggu 2-5 menit hingga aplikasi selesai di-deploy."
echo "Anda akan mendapat URL publik seperti:"
echo "   https://$REPO_NAME-lowercase.streamlit.app/"
echo ""
echo "Bagikan URL tersebut ke dosen!"
echo "============================================"
