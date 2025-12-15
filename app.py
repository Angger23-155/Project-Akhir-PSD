import streamlit as st
import joblib
import os
import numpy as np
import librosa
from PIL import Image

# --- KONFIGURASI PATH ---
WORKDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'model_pipeline_selectk.joblib'
MODEL_PATH = os.path.join(WORKDIR, MODEL_NAME)
DATA_PATH = os.path.join(WORKDIR, 'data_processed.npz')
FIG_DIR = os.path.join(WORKDIR, 'figures')

st.set_page_config(page_title="Cats vs Dogs Audio Classifier", layout="wide")

# --- SIDEBAR: METRICS INFO ---
st.sidebar.title("📊 Performa Model")
st.sidebar.markdown("Statistik evaluasi pada data uji (20% Split):")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Akurasi", "63.6%")
with col2:
    st.metric("F1-Macro", "0.636")

st.sidebar.markdown("---")
st.sidebar.markdown("**Validasi Silang (Nested CV):**")
st.sidebar.info("F1-Macro: 0.525 ± 0.058")

st.sidebar.markdown("---")
st.sidebar.markdown("**Detail Pipeline:**")
st.sidebar.code("VarianceThreshold -> StandardScaler -> SelectKBest(k=200) -> RandomForest(n=100)")
st.sidebar.caption("Project Akhir PSD - Angger Maulana Effendi")

# --- HEADER UTAMA ---
st.title('🎵 Klasifikasi Audio Cats vs Dogs')
st.markdown(f'Model: **SelectKBest + RandomForest** | Dataset: **CatsDogs TimeSeries**')

# --- FUNGSI BANTUAN (UPDATED) ---
def process_audio_file(uploaded_file, target_length=14773):
    """
    Memproses audio dengan:
    1. Resample ke 8000Hz (agar cakupan waktu per sample lebih luas)
    2. Trimming (menghapus hening di awal/akhir)
    3. Padding/Cutting ke 14773 fitur
    """
    try:
        # 1. Load dengan SR 8000Hz (PENTING: Menyesuaikan dataset TimeSeries umum)
        y, sr = librosa.load(uploaded_file, sr=8000)
        
        # 2. Hapus Hening (Silence Trimming)
        # Menghapus bagian diam (noise floor < 20db) di awal dan akhir
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # 3. Logika Padding/Truncating
        if len(y) > target_length:
            # Jika masih terlalu panjang setelah di-trim, ambil bagian tengah
            # (Bagian tengah biasanya berisi informasi suara utama)
            start = (len(y) - target_length) // 2
            y = y[start : start + target_length]
        else:
            # Jika kurang, tambahkan nol (padding constant) di belakang
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), 'constant')
            
        return y.reshape(1, -1)
    except Exception as e:
        st.error(f"Error memproses audio: {e}")
        return None

def decode_label(x):
    try:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except:
        return str(x)

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f'❌ File model `{MODEL_NAME}` tidak ditemukan.')
    st.info("Pastikan file model sudah di-upload ke GitHub sejajar dengan app.py")
    st.stop()
else:
    model = joblib.load(MODEL_PATH)
    # Ambil jumlah fitur yang diharapkan model (default 14773)
    expected_features = getattr(model, 'n_features_in_', 14773)

# --- TABS UTAMA ---
tab1, tab2, tab3 = st.tabs(["📂 Gunakan Test Set", "🎙️ Upload File .wav", "📊 Visualisasi Data"])

# === TAB 1: DATA TEST (Internal) ===
with tab1:
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        st.info("Simulasi prediksi menggunakan 20% data test yang sudah dipisahkan sebelumnya.")
        
        col_main, col_stat = st.columns([2, 1])
        with col_main:
            idx = st.slider('Pilih Index Sample Test Set', 0, len(X_test)-1, 0)
            sample = X_test[idx].reshape(1, -1)
            true_lbl = decode_label(y_test[idx])
            
            if st.button('🔍 Prediksi Sample Ini'):
                pred = model.predict(sample)[0]
                pred_lbl = decode_label(pred)
                
                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Label Sebenarnya", true_lbl)
                
                if true_lbl.lower() == pred_lbl.lower():
                    res_c2.success(f"✅ Prediksi Benar: {pred_lbl}")
                else:
                    res_c2.error(f"❌ Prediksi Salah: {pred_lbl}")
        
        with col_stat:
            st.write(f"**Total Sample Test:** {len(X_test)}")
            st.write(f"**Dimensi Fitur:** {X_test.shape[1]}")

    else:
        st.warning("Data test tidak ditemukan.")

# === TAB 2: UPLOAD AUDIO (Eksternal) ===
with tab2:
    st.write("Uji coba model dengan file audio eksternal (.wav).")
    uploaded_wav = st.file_uploader("Upload file .wav di sini", type=["wav"])
    
    if uploaded_wav is not None:
        st.audio(uploaded_wav)
        if st.button("⚡ Analisis Audio"):
            with st.spinner("Sedang memproses (Resampling 8kHz & Trimming)..."):
                feats = process_audio_file(uploaded_wav, target_length=expected_features)
                if feats is not None:
                    pred = model.predict(feats)[0]
                    lbl = decode_label(pred)
                    
                    st.markdown("### Hasil Klasifikasi:")
                    if 'cat' in lbl.lower() or 'kucing' in lbl.lower():
                        st.success(f"🐱 **KUCING (Cat)**")
                    else:
                        st.success(f"🐶 **ANJING (Dog)**")

# === TAB 3: VISUALISASI (Laporan) ===
with tab3:
    st.header("🖼️ Visualisasi & Analisis")
    
    # 1. Confusion Matrix
    st.subheader("1. Evaluasi Model (Confusion Matrix)")
    cm_path = os.path.join(FIG_DIR, 'confusion_matrix_selectk_best.png')
    
    if os.path.exists(cm_path):
        col_cm1, col_cm2 = st.columns([1, 2])
        with col_cm1:
            st.markdown("""
            **Interpretasi:**
            - **Diagonal**: Jumlah prediksi yang **Benar**.
            - **Off-Diagonal**: Jumlah prediksi yang **Salah**.
            Gambar ini menunjukkan seberapa baik model membedakan suara Kucing vs Anjing pada data uji.
            """)
        with col_cm2:
            st.image(Image.open(cm_path), caption="Confusion Matrix: SelectKBest + RF", use_column_width=True)
    else:
        st.warning("Gambar Confusion Matrix tidak ditemukan.")

    st.markdown("---")

    # 2. Preprocessing
    st.subheader("2. Proses Preprocessing Data")
    st.write("Visualisasi langkah-langkah penyiapan data:")
    
    # Sesuaikan nama file gambar di sini
    prep_images = [
        {"name": "Distribusi Kelas", "file": "class_distribution.png"},
        {"name": "Histogram Variansi", "file": "variance_histogram.png"},
        {"name": "Sampel Boxplot", "file": "boxplots_first4.png"}
    ]
    
    cols = st.columns(3)
    for i, item in enumerate(prep_images):
        with cols[i % 3]:
            img_p = os.path.join(FIG_DIR, item["file"])
            if os.path.exists(img_p):
                st.image(Image.open(img_p), caption=item["name"], use_column_width=True)
            else:
                st.info(f"Gambar {item['name']} tidak tersedia.")