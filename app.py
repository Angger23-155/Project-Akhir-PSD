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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI PREPROCESSING AUDIO (LOGIKA 16kHz) ---
def process_audio_file(uploaded_file, target_length=14773):
    """
    Preprocessing Audio agar sesuai dengan dataset CatsDogs (TSC):
    1. Resample ke 16.000 Hz (Standar Dataset TSC).
    2. Trim Silence (Hapus hening di awal).
    3. Normalisasi Amplitudo.
    4. Padding/Looping agar pas 14773 fitur.
    """
    try:
        # 1. Load dengan SR 16000 (KUNCI UTAMA AKURASI)
        # Dataset CatsDogs TSC dilatih pada 16kHz.
        y, sr = librosa.load(uploaded_file, sr=16000)
        
        # 2. Hapus Hening (Silence Trimming)
        # Ambang batas 20dB untuk membuang noise ruangan di awal
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Gunakan hasil trim jika tidak kosong, jika kosong (hening total) pakai aslinya
        if len(y_trimmed) > 0:
            y = y_trimmed
            
        # 3. Normalisasi Amplitudo (Agar volume input = volume training)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        # 4. Fitur Engineering: Padding / Cutting
        if len(y) < target_length:
            # Jika terlalu pendek: Ulangi suara (Looping) sampai penuh
            # Ini lebih baik dari padding nol (diam)
            n_repeat = int(np.ceil(target_length / len(y)))
            y = np.tile(y, n_repeat)
            y = y[:target_length]
        else:
            # Jika terlalu panjang: Ambil bagian paling keras (energi tertinggi)
            # Karena suara kucing/anjing biasanya singkat dan meledak
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            max_rms_idx = np.argmax(rms)
            # Konversi frame index ke sample index
            peak_sample = max_rms_idx * 512
            
            # Tentukan start dan end di sekitar puncak suara
            start = max(0, peak_sample - (target_length // 2))
            end = start + target_length
            
            # Koreksi jika melebihi batas
            if end > len(y):
                end = len(y)
                start = max(0, end - target_length)
                
            y = y[start:end]
            
        return y.reshape(1, -1)
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return None

def decode_label(x):
    try:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except:
        return str(x)

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f'❌ Model tidak ditemukan: {MODEL_NAME}')
    st.info("Pastikan file model.joblib sudah di-push ke GitHub.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- INTERFACE ---
st.title('🎵 Klasifikasi Audio Cats vs Dogs')
st.caption('Model: SelectKBest + RandomForest | Sampling Rate: 16.000 Hz | Input: 14.773 Fitur')

# TABS
tab1, tab2, tab3 = st.tabs(["📂 Test Set (Internal)", "🎙️ Upload .wav (Live)", "📊 Visualisasi"])

# === TAB 1: DATA TEST ===
with tab1:
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("Prediksi menggunakan 20% data test yang sudah dipisahkan (Split).")
            idx = st.slider('Pilih Index Sample', 0, len(X_test)-1, 0)
            
            if st.button('🔍 Prediksi Sample Internal'):
                sample = X_test[idx].reshape(1, -1)
                pred = model.predict(sample)[0]
                true_lbl = decode_label(y_test[idx])
                pred_lbl = decode_label(pred)
                
                c1, c2 = st.columns(2)
                c1.metric("Label Sebenarnya", true_lbl.upper())
                if true_lbl.lower() == pred_lbl.lower():
                    c2.success(f"✅ Prediksi: {pred_lbl.upper()}")
                else:
                    c2.error(f"❌ Prediksi: {pred_lbl.upper()}")
        with col2:
            st.write(f"**Total Sample:** {len(X_test)}")
            st.write(f"**Fitur:** {X_test.shape[1]}")
    else:
        st.warning("File data_processed.npz tidak ditemukan.")

# === TAB 2: UPLOAD ===
with tab2:
    st.write("### 🎙️ Uji Coba Audio Sendiri")
    st.markdown("Agar akurat, gunakan rekaman yang jelas (minim noise) dan berdurasi sekitar 1 detik.")
    
    uploaded_wav = st.file_uploader("Upload file .wav", type=["wav"])
    
    if uploaded_wav:
        st.audio(uploaded_wav)
        
        if st.button("⚡ Analisis Audio"):
            with st.spinner("Memproses audio (Resampling 16kHz, Trimming, Normalizing)..."):
                # Proses audio
                feats = process_audio_file(uploaded_wav)
                
                if feats is not None:
                    # Prediksi Probabilitas
                    try:
                        probs = model.predict_proba(feats)[0] # [Prob_Cat, Prob_Dog]
                        # Asumsi kelas: 0 = Cat, 1 = Dog (berdasarkan sidebar kamu)
                        p_cat = probs[0]
                        p_dog = probs[1]
                        
                        # Tampilkan Progress Bar
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.markdown(f"**KUCING (Cat)**: `{p_cat:.1%}`")
                            st.progress(float(p_cat))
                        with col_res2:
                            st.markdown(f"**ANJING (Dog)**: `{p_dog:.1%}`")
                            st.progress(float(p_dog))
                        
                        # Keputusan Final
                        st.markdown("---")
                        if p_cat > p_dog:
                            st.success(f"🐱 Hasil Prediksi: **KUCING** ({p_cat:.1%})")
                        else:
                            st.success(f"🐶 Hasil Prediksi: **ANJING** ({p_dog:.1%})")
                            
                    except:
                        # Fallback jika model tidak support predict_proba
                        pred = model.predict(feats)[0]
                        lbl = decode_label(pred)
                        st.info(f"Hasil Prediksi: {lbl}")

# === TAB 3: VISUALISASI ===
with tab3:
    st.header("🖼️ Visualisasi Model & Data")
    
    # Confusion Matrix
    cm_path = os.path.join(FIG_DIR, 'confusion_matrix_selectk_best.png')
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path), caption="Confusion Matrix Model", width=500)
    else:
        st.warning("Confusion matrix tidak ditemukan di folder figures.")
        
    st.markdown("---")
    st.subheader("Distribusi & Preprocessing")
    
    # Galeri Gambar
    # Pastikan nama file ini ada di folder figures kamu!
    img_list = ["class_distribution.png", "variance_histogram.png", "boxplots_first4.png"]
    
    cols = st.columns(3)
    for i, fname in enumerate(img_list):
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            with cols[i%3]:
                st.image(Image.open(fpath), caption=fname, use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Project Akhir PSD - Angger Maulana Effendi")