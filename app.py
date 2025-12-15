import streamlit as st
import joblib
import os
import numpy as np
import librosa
from PIL import Image

# --- KONFIGURASI PATH ---
# Menggunakan path relatif agar aman di Local & Cloud
WORKDIR = os.path.dirname(os.path.abspath(__file__))

# Sesuaikan nama file model dengan yang kamu sebutkan
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
    """
    Mengubah file audio .wav menjadi array 1D dengan panjang tetap
    sesuai format dataset CatsDogs (default: 14773 fitur).
    """
    try:
        # 1. Load audio dengan librosa (resample ke default sr jika perlu, atau gunakan sr asli)
        # Kita gunakan sr=None untuk mempertahankan sampling rate asli, 
        # namun dataset aslinya mungkin punya sr spesifik.
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 2. Sesuaikan panjang data (Padding atau Truncating)
        if len(y) > target_length:
            # Jika terlalu panjang, potong
            y = y[:target_length]
        else:
            # Jika terlalu pendek, tambahkan nol (padding) di belakang
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), 'constant')
            
        # 3. Reshape ke format (1, n_features) untuk prediksi model
        return y.reshape(1, -1)
    except Exception as e:
        st.error(f"Error memproses audio: {e}")
        return None

def decode_label(x):
    """Mendecode label dari byte/string"""
    try:
        if isinstance(x, (bytes, bytearray)):
            return x.decode()
        return str(x)
    except:
        return str(x)

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f'❌ File model `{MODEL_NAME}` tidak ditemukan di: {WORKDIR}')
    st.info("Pastikan file model sudah di-upload ke GitHub sejajar dengan app.py")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    # Cek apakah model punya atribut n_features_in_ (untuk validasi panjang input)
    expected_features = getattr(model, 'n_features_in_', 14773) 
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- INTERFACE UTAMA ---
# Tab untuk memisahkan fitur
tab1, tab2 = st.tabs(["📂 Gunakan Test Set (20%)", "🎙️ Upload File .wav"])

# === TAB 1: PREDIKSI DARI DATASET ===
with tab1:
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        st.info("Prediksi menggunakan data test yang sudah displit (20%) dari dataset asli.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            idx = st.slider('Pilih Index Sample Test Set', 0, len(X_test)-1, 0)
            
            # Ambil data
            sample_data = X_test[idx].reshape(1, -1)
            true_lbl = decode_label(y_test[idx])
            
            if st.button('🔍 Prediksi Sample Ini'):
                pred = model.predict(sample_data)[0]
                pred_lbl = decode_label(pred)
                
                # Tampilkan Hasil
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Label Sebenarnya", true_lbl)
                with res_col2:
                    is_correct = (true_lbl.lower() == pred_lbl.lower())
                    if is_correct:
                        st.success(f"Prediksi: {pred_lbl} ✅")
                    else:
                        st.error(f"Prediksi: {pred_lbl} ❌")
                        
        with col2:
            st.markdown("### Statistik Data")
            st.write(f"- Jumlah Test Sample: **{len(X_test)}**")
            st.write(f"- Fitur per Sample: **{sample_data.shape[1]}**")
            
            # Tampilkan Gambar Confusion Matrix jika ada
            cm_path = os.path.join(FIG_DIR, 'confusion_matrix.png') # Sesuaikan nama file gambar
            if os.path.exists(cm_path):
                st.image(Image.open(cm_path), caption="Confusion Matrix", use_column_width=True)
                
    else:
        st.warning(f"File dataset `{DATA_PATH}` tidak ditemukan. Pastikan file ada di GitHub.")

# === TAB 2: UPLOAD WAV ===
with tab2:
    st.info("⚠️ Catatan: Model dilatih dengan dataset spesifik (CatsDogs TSC). Rekaman audio acak mungkin memerlukan preprocessing (trimming/denoising) agar akurat.")
    
    uploaded_wav = st.file_uploader("Upload file suara (.wav)", type=["wav"])
    
    if uploaded_wav is not None:
        st.audio(uploaded_wav, format='audio/wav')
        
        if st.button("⚡ Analisis Audio Upload"):
            with st.spinner("Memproses audio..."):
                # Proses audio menjadi array
                # Kita gunakan expected_features dari model agar panjangnya pas
                features = process_audio_file(uploaded_wav, target_length=expected_features)
                
                if features is not None:
                    try:
                        # Lakukan Prediksi
                        prediction = model.predict(features)[0]
                        pred_label = decode_label(prediction)
                        
                        st.markdown("---")
                        st.subheader("Hasil Prediksi:")
                        
                        # Tampilan hasil yang menarik
                        if 'cat' in pred_label.lower() or 'kucing' in pred_label.lower():
                            st.success(f"🐱 **Meong! Ini terdeteksi sebagai KUCING (Cat)**")
                        elif 'dog' in pred_label.lower() or 'anjing' in pred_label.lower():
                            st.warning(f"🐶 **Guk! Ini terdeteksi sebagai ANJING (Dog)**")
                        else:
                            st.info(f"🏷️ Kelas terdeteksi: {pred_label}")
                            
                    except Exception as e:
                        st.error(f"Gagal memprediksi: {str(e)}")
                        st.caption("Kemungkinan format audio terlalu berbeda dengan data training.")

st.markdown("---")
st.caption("Project Akhir PSD | Deployed with Streamlit")