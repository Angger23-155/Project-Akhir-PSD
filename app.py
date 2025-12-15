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

# --- CSS STYLING ---
st.markdown("""
<style>
    .result-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 10px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI SMART SCANNING (SOLUSI AKURASI) ---
def process_audio_scanning(uploaded_file, target_length=14773):
    """
    Menghasilkan 3 variasi alignment (Kiri, Tengah, Kanan)
    untuk mengatasi masalah pergeseran posisi suara pada model Random Forest.
    """
    try:
        # 1. Load Audio (16kHz standard TSC)
        y, sr = librosa.load(uploaded_file, sr=16000)
        
        # 2. Trim Silence (Hapus hening)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        # 3. Normalisasi
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 4. Generate 3 Kandidat (Batch Processing)
        candidates = []
        
        # Jika audio lebih PENDEK dari target, kita padding dengan 3 cara
        if len(y) < target_length:
            pad_needed = target_length - len(y)
            
            # Kandidat 1: Left Align (Padding di Kanan) -> Suara di Awal
            c1 = np.pad(y, (0, pad_needed), 'constant')
            
            # Kandidat 2: Center Align (Padding Kiri Kanan) -> Suara di Tengah
            pad_l = pad_needed // 2
            pad_r = pad_needed - pad_l
            c2 = np.pad(y, (pad_l, pad_r), 'constant')
            
            # Kandidat 3: Right Align (Padding di Kiri) -> Suara di Akhir
            # (Ini penting karena file ARFF kamu banyak nol di awal!)
            c3 = np.pad(y, (pad_needed, 0), 'constant')
            
            candidates = [c1, c2, c3]
            
        # Jika audio lebih PANJANG, kita potong dengan 3 cara
        else:
            # Kandidat 1: Awal
            c1 = y[:target_length]
            
            # Kandidat 2: Tengah
            start_mid = (len(y) - target_length) // 2
            c2 = y[start_mid : start_mid + target_length]
            
            # Kandidat 3: Akhir
            c3 = y[-target_length:]
            
            candidates = [c1, c2, c3]

        # Stack menjadi array (3, 14773)
        return np.array(candidates)

    except Exception as e:
        st.error(f"Error processing: {e}")
        return None

def decode_label(x):
    try:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except:
        return str(x)

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f'❌ Model `{MODEL_NAME}` tidak ditemukan.')
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# --- MAIN UI ---
st.title('🎵 Klasifikasi Audio Cats vs Dogs')
st.caption('Model: SelectKBest + RandomForest | Strategy: Multi-Alignment Scanning')

# TABS
tab1, tab2, tab3 = st.tabs(["📂 Test Set (Internal)", "🎙️ Upload .wav (Live)", "📊 Visualisasi"])

# === TAB 1: DATA TEST ===
with tab1:
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        
        st.info("Prediksi Data Test (20% Split)")
        col1, col2 = st.columns([2, 1])
        with col1:
            idx = st.slider('Pilih Index', 0, len(X_test)-1, 0)
            if st.button('Prediksi'):
                sample = X_test[idx].reshape(1, -1)
                pred = model.predict(sample)[0]
                true_lbl = decode_label(y_test[idx])
                pred_lbl = decode_label(pred)
                
                c1, c2 = st.columns(2)
                c1.metric("Asli", true_lbl.upper())
                if true_lbl.lower() == pred_lbl.lower():
                    c2.success(f"Prediksi: {pred_lbl.upper()} ✅")
                else:
                    c2.error(f"Prediksi: {pred_lbl.upper()} ❌")
    else:
        st.warning("Data test tidak ditemukan.")

# === TAB 2: UPLOAD (DENGAN SCANNING) ===
with tab2:
    st.write("### 🎙️ Uji Coba Audio Sendiri")
    st.markdown("Menggunakan teknik **Scanning** (mencoba posisi Awal, Tengah, Akhir) untuk akurasi maksimal.")
    
    uploaded_wav = st.file_uploader("Upload .wav", type=["wav"])
    
    if uploaded_wav:
        st.audio(uploaded_wav)
        
        if st.button("⚡ Analisis Smart Scan"):
            with st.spinner("Scanning alignment terbaik..."):
                # Dapatkan 3 versi audio
                X_batch = process_audio_scanning(uploaded_wav)
                
                if X_batch is not None:
                    try:
                        # Prediksi Probabilitas untuk ketiganya sekaligus
                        # Output shape: (3, 2) -> [ [prob_cat, prob_dog], ... ]
                        all_probs = model.predict_proba(X_batch)
                        
                        # Kita ambil RATA-RATA Probabilitas (Soft Voting)
                        # Ini membuat prediksi lebih stabil
                        avg_probs = np.mean(all_probs, axis=0)
                        
                        # Probabilitas Final
                        p_cat = avg_probs[0] # Asumsi index 0 = Cat
                        p_dog = avg_probs[1] # Asumsi index 1 = Dog
                        
                        # Cek Label Classes (Kadang terbalik tergantung training)
                        # Kita asumsikan default alfabet: 0=Cat, 1=Dog. 
                        # Jika di modelmu terbalik, tukar variabel p_cat dan p_dog di sini.
                        if hasattr(model, 'classes_'):
                            classes = model.classes_
                            # Logika mapping sederhana
                            if 'dog' in str(classes[0]).lower():
                                p_dog, p_cat = avg_probs[0], avg_probs[1]
                        
                        # TAMPILAN HASIL
                        st.markdown("---")
                        c1, c2 = st.columns(2)
                        
                        with c1:
                            st.markdown(f"<h3 style='text-align: center;'>🐱 Kucing</h3>", unsafe_allow_html=True)
                            st.progress(float(p_cat))
                            st.markdown(f"<div style='text-align: center; font-weight:bold;'>{p_cat:.1%}</div>", unsafe_allow_html=True)
                            
                        with c2:
                            st.markdown(f"<h3 style='text-align: center;'>🐶 Anjing</h3>", unsafe_allow_html=True)
                            st.progress(float(p_dog))
                            st.markdown(f"<div style='text-align: center; font-weight:bold;'>{p_dog:.1%}</div>", unsafe_allow_html=True)
                        
                        # KEPUTUSAN FINAL
                        st.markdown("---")
                        final_pred = "KUCING" if p_cat > p_dog else "ANJING"
                        confidence = max(p_cat, p_dog)
                        
                        if final_pred == "KUCING":
                            st.success(f"🎉 **Hasil Akhir: {final_pred}** (Keyakinan: {confidence:.1%})")
                        else:
                            st.success(f"🎉 **Hasil Akhir: {final_pred}** (Keyakinan: {confidence:.1%})")
                            
                        # DEBUG: Tampilkan detail scan (Optional)
                        with st.expander("Lihat Detail Scanning"):
                            st.write("Model menguji 3 posisi suara:")
                            st.write(f"1. Posisi Awal (Kiri): Cat {all_probs[0][0]:.2f} | Dog {all_probs[0][1]:.2f}")
                            st.write(f"2. Posisi Tengah: Cat {all_probs[1][0]:.2f} | Dog {all_probs[1][1]:.2f}")
                            st.write(f"3. Posisi Akhir (Kanan): Cat {all_probs[2][0]:.2f} | Dog {all_probs[2][1]:.2f}")
                            
                    except Exception as e:
                        st.error(f"Error prediksi: {e}")
                        # Fallback ke prediksi biasa
                        pred = model.predict(X_batch)[0]
                        st.info(f"Prediksi Raw: {pred}")

# === TAB 3: VISUALISASI ===
with tab3:
    st.header("🖼️ Visualisasi Project")
    
    # Confusion Matrix
    cm_path = os.path.join(FIG_DIR, 'confusion_matrix_selectk_best.png')
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path), caption="Confusion Matrix", width=500)
    
    # Gallery
    st.subheader("Preprocessing Insight")
    img_list = ["class_distribution.png", "variance_histogram.png", "boxplots_first4.png"]
    cols = st.columns(3)
    for i, fname in enumerate(img_list):
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            with cols[i%3]:
                st.image(Image.open(fpath), caption=fname, use_column_width=True)