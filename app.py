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

# --- HEADER ---
st.title('🎵 Klasifikasi Audio Cats vs Dogs')
st.markdown(f'Model: **SelectKBest + RandomForest** | Dataset: **CatsDogs TimeSeries**')

# --- FUNGSI BANTUAN ---
def process_audio_file(uploaded_file, target_length=14773):
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        if len(y) > target_length:
            y = y[:target_length]
        else:
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
    expected_features = getattr(model, 'n_features_in_', 14773)

# --- TABS UTAMA ---
tab1, tab2, tab3 = st.tabs(["📂 Gunakan Test Set", "🎙️ Upload File .wav", "📊 Visualisasi Data"])

# === TAB 1: DATA TEST ===
with tab1:
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        st.info("Prediksi menggunakan data test (20% split).")
        col1, col2 = st.columns([2, 1])
        with col1:
            idx = st.slider('Pilih Index Sample', 0, len(X_test)-1, 0)
            sample = X_test[idx].reshape(1, -1)
            true_lbl = decode_label(y_test[idx])
            
            if st.button('🔍 Prediksi Sample Ini'):
                pred = model.predict(sample)[0]
                pred_lbl = decode_label(pred)
                
                c1, c2 = st.columns(2)
                c1.metric("Label Asli", true_lbl)
                if true_lbl.lower() == pred_lbl.lower():
                    c2.success(f"✅ Prediksi: {pred_lbl}")
                else:
                    c2.error(f"❌ Prediksi: {pred_lbl}")
        with col2:
             st.metric("Total Test Samples", len(X_test))
    else:
        st.warning("Data test tidak ditemukan.")

# === TAB 2: UPLOAD AUDIO ===
with tab2:
    st.write("Uji coba dengan file audio eksternal.")
    uploaded_wav = st.file_uploader("Upload .wav", type=["wav"])
    if uploaded_wav is not None:
        st.audio(uploaded_wav)
        if st.button("⚡ Analisis Audio"):
            with st.spinner("Memproses..."):
                feats = process_audio_file(uploaded_wav, target_length=expected_features)
                if feats is not None:
                    pred = model.predict(feats)[0]
                    lbl = decode_label(pred)
                    if 'cat' in lbl.lower() or 'kucing' in lbl.lower():
                        st.success(f"🐱 Hasil: **KUCING (Cat)**")
                    else:
                        st.success(f"🐶 Hasil: **ANJING (Dog)**")

# === TAB 3: VISUALISASI ===
with tab3:
    st.header("🖼️ Galeri Visualisasi Project")
    
    # 1. Confusion Matrix (Paling Penting)
    st.subheader("1. Evaluasi Model (Confusion Matrix)")
    cm_file = 'confusion_matrix_selectk_best.png'
    cm_path = os.path.join(FIG_DIR, cm_file)
    
    if os.path.exists(cm_path):
        # Tampilkan besar di tengah
        col_cm1, col_cm2 = st.columns([1, 2])
        with col_cm1:
             st.write("""
             **Analisis:**
             Confusion matrix menunjukkan performa model pada data uji. 
             Diagonal utama menunjukkan prediksi yang benar.
             """)
        with col_cm2:
             st.image(Image.open(cm_path), caption="SelectKBest + Random Forest", use_column_width=True)
    else:
        st.warning(f"File `{cm_file}` tidak ditemukan di folder figures.")

    st.markdown("---")

    # 2. Gambar Preprocessing (Grid Layout)
    st.subheader("2. Insight Data & Preprocessing")
    
    # Daftar nama file sesuai screenshot kamu
    # Pastikan ekstensi filenya benar (.png atau .jpg)
    prep_images = [
        {"file": "class_distribution.png", "caption": "Distribusi Kelas (Seimbang/Tidak)"},
        {"file": "variance_histogram.png", "caption": "Histogram Variansi Fitur"},
        {"file": "boxplots_first4.png", "caption": "Boxplot 4 Fitur Pertama"}
    ]
    
    # Tampilkan dalam kolom
    cols = st.columns(3)
    for i, item in enumerate(prep_images):
        with cols[i % 3]: # Loop kolom agar rapi
            img_path = os.path.join(FIG_DIR, item["file"])
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=item["caption"], use_column_width=True)
            else:
                st.info(f"Gambar `{item['file']}` belum tersedia.")

st.caption("Project Akhir PSD - Angger Maulana Effendi")