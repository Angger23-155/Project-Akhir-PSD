import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

WORKDIR = r"d:\Project Akhir PSD"
MODEL_PATH = os.path.join(WORKDIR, 'model_pipeline.joblib')
DATA_PATH = os.path.join(WORKDIR, 'data_processed.npz')
import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Optional audio feature extraction
try:
    import librosa
except Exception:
    librosa = None

WORKDIR = r"d:\Project Akhir PSD"
MODEL_PATH = os.path.join(WORKDIR, 'model_pipeline.joblib')
DATA_PATH = os.path.join(WORKDIR, 'data_processed.npz')
FIG_DIR = os.path.join(WORKDIR, 'figures')

st.title('Demo Klasifikasi Cats vs Dogs - Model Pipeline')

def decode_label(x):
    try:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except Exception:
        return str(x)

if not os.path.exists(MODEL_PATH):
    st.error(f'Model tidak ditemukan di {MODEL_PATH}. Jalankan notebook untuk menyimpan model terlebih dahulu.')
    st.stop()

# Muat model
model = joblib.load(MODEL_PATH)
st.success('Model dimuat: ' + os.path.basename(MODEL_PATH))

col1, col2 = st.columns([2,1])

with col1:
    st.header('Prediksi dari test set (contoh)')
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        label_classes = data['label_classes']

        idx = st.slider('Index contoh test', 0, max(0, X_test.shape[0]-1), 0)
        sample = X_test[idx:idx+1]
        try:
            pred = model.predict(sample)[0]
        except Exception as e:
            st.error('Gagal memprediksi sample test: ' + str(e))
            pred = None

        st.write('Label sebenarnya:', decode_label(y_test[idx]))
        st.write('Prediksi model:', decode_label(pred))

        cm_path = os.path.join(FIG_DIR, 'confusion_matrix_selectk_best.png')
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption='Confusion Matrix (SelectKBest)')
        else:
            st.info('Tidak menemukan gambar confusion matrix di figures/')
    else:
        st.info('Data test tidak ditemukan (data_processed.npz).')

with col2:
    st.header('Upload file .wav untuk prediksi (experimental)')
    st.write('Catatan: model ini dilatih pada fitur yang diekstrak sebelumnya. Prediksi dari file .wav akan menggunakan MFCC ringkasan (mean+std per koefisien). Jika dimensi fitur tidak cocok dengan model, prediksi mungkin gagal atau tidak akurat.')
    uploaded = st.file_uploader('Pilih file .wav', type=['wav', 'mp3'])
    if uploaded is not None:
        if librosa is None:
            st.error('Librosa tidak tersedia. Instal dependency `librosa` untuk menggunakan fitur upload audio.')
        else:
            # Simpan sementara dan proses
            tmp_path = os.path.join(WORKDIR, 'tmp_upload.wav')
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            try:
                y, sr = librosa.load(tmp_path, sr=22050)
                # Ekstrak MFCC dan ringkasan statistik (mean, std)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfcc_mean = mfcc.mean(axis=1)
                mfcc_std = mfcc.std(axis=1)
                feat = np.concatenate([mfcc_mean, mfcc_std])  # vector length 80
                st.write('Ekstrak MFCC selesai. Dimensi fitur:', feat.shape)

                # Siapkan input untuk model
                X_in = np.atleast_2d(feat)
                # Coba prediksi; jika bentuk berbeda, tangani exception dan tampilkan info
                try:
                    pred = model.predict(X_in)[0]
                    st.success('Prediksi: ' + decode_label(pred))
                except Exception as e:
                    # Coba informasikan ukuran fitur yang diharapkan jika tersedia
                    expected = getattr(model, 'n_features_in_', None)
                    if expected is None:
                        # coba lihat langkah terakhir (classifier) jika tersedia
                        try:
                            expected = model.named_steps['clf'].n_features_in_
                        except Exception:
                            expected = None
                    st.error('Gagal memprediksi dari audio: ' + str(e))
                    if expected is not None:
                        st.info(f'Model mengharapkan input berdimensi {expected}, sementara fitur audio menyediakan {feat.size}.')
                    else:
                        st.info(f'Fitur audio berdimensi {feat.size}. Model dan fitur tidak kompatibel.')
            except Exception as e:
                st.error('Gagal ekstraksi audio: ' + str(e))

st.markdown('---')
st.write('Model dan demo ini dibuat untuk tujuan presentasi. Untuk produksi, lakukan validasi lebih lanjut dan pastikan pipeline ekstraksi fitur sesuai antara data latih dan input runtime.')
