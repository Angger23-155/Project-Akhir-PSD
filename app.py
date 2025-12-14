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
from PIL import Image

# Gunakan path relatif agar berfungsi di lokal dan cloud
WORKDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKDIR, 'model_pipeline.joblib')
DATA_PATH = os.path.join(WORKDIR, 'data_processed.npz')
FIG_DIR = os.path.join(WORKDIR, 'figures')

st.set_page_config(page_title="Cats vs Dogs Classifier", layout="wide")
st.title('🎵 Demo Klasifikasi Audio Cats vs Dogs')
st.markdown('Menggunakan Pipeline SelectKBest + RandomForest')

def decode_label(x):
    try:
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except Exception:
        return str(x)

if not os.path.exists(MODEL_PATH):
    st.error(f'❌ Model tidak ditemukan. Pastikan file `model_pipeline.joblib` ada di repository.')
    st.stop()

# Muat model
model = joblib.load(MODEL_PATH)
st.success('✅ Model dimuat: SelectKBest + RandomForestClassifier')

# Sidebar: Informasi performa
st.sidebar.markdown('## 📊 Performa Model')
st.sidebar.markdown('''
- **Test Accuracy**: 0.636 (63.6%)
- **Test F1-macro**: 0.636
- **CV F1-macro** (nested): 0.525 ± 0.058
- **Pipeline**: VarianceThreshold → StandardScaler → SelectKBest(k=200) → RF(n_estimators=100)
- **Dataset**: ~130 training samples, 33 test samples
''')

st.sidebar.markdown('---')
st.sidebar.markdown('## 📁 Repository')
st.sidebar.markdown('[GitHub: Project-Akhir-PSD](https://github.com/Angger23-155/Project-Akhir-PSD)')
st.sidebar.markdown('[Demo: Streamlit Cloud](https://klasifikasi-catsdogs.streamlit.app/)')

# Main content
if os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    label_classes = data['label_classes']

    st.markdown('## 🎯 Prediksi Contoh dari Test Set')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        idx = st.slider(
            'Pilih contoh test (geser slider untuk lihat prediksi berbeda)',
            0, 
            max(0, X_test.shape[0]-1), 
            0
        )
        
        sample = X_test[idx:idx+1]
        try:
            pred = model.predict(sample)[0]
            true_label = decode_label(y_test[idx])
            pred_label = decode_label(pred)
            
            # Tampilkan hasil
            st.write(f'**Sampel test index**: {idx}')
            
            col_true, col_pred = st.columns(2)
            with col_true:
                st.info(f'🏷️ **Label Sebenarnya**: `{true_label.upper()}`')
            with col_pred:
                if true_label.lower() == pred_label.lower():
                    st.success(f'✅ **Prediksi**: `{pred_label.upper()}` (BENAR)')
                else:
                    st.warning(f'❌ **Prediksi**: `{pred_label.upper()}` (SALAH)')
        except Exception as e:
            st.error(f'Gagal memprediksi: {str(e)}')

    with col2:
        st.write('')
        st.write('')
        st.markdown('**Statistik**')
        st.metric('Total Test Samples', X_test.shape[0])
        st.metric('Fitur per Sample', X_test.shape[1])

    # Confusion Matrix
    st.markdown('---')
    st.markdown('## 📈 Confusion Matrix (Test Set)')
    cm_path = os.path.join(FIG_DIR, 'confusion_matrix_selectk_best.png')
    if os.path.exists(cm_path):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(Image.open(cm_path), caption='SelectKBest + RandomForest', use_column_width=True)
        with col2:
            st.markdown('''
            **Interpretasi**:
            - **Diagonal**: Prediksi benar
            - **Off-diagonal**: Prediksi salah
            - **Label 0**: Cats (Kucing)
            - **Label 1**: Dogs (Anjing)
            ''')
    else:
        st.warning('Gambar confusion matrix tidak ditemukan.')

else:
    st.error('❌ Data test tidak ditemukan (`data_processed.npz`).')

st.markdown('---')
st.markdown('''
**Catatan**:
- Model dilatih pada dataset CatsDogs dengan ekstraksi fitur time-series yang telah diproses
- Fitur seleksi: SelectKBest memilih 200 fitur terbaik dari ribuan fitur original
- Untuk akurasi lebih tinggi, diperlukan: augmentasi data, ekstraksi fitur lebih baik (MFCC), atau lebih banyak sampel training
- Demo ini dibuat untuk tujuan akademik (presentasi tugas akhir)

**Referensi**:
- Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks
- Salamon, J., et al. (2014). Dataset and Taxonomy for Urban Sound Research
''')
