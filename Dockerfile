FROM python:3.10-slim

WORKDIR /app

# Salin requirements dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin aplikasi dan file model
COPY app.py .
COPY model_pipeline.joblib .
COPY data_processed.npz .
COPY figures/ ./figures/
COPY .streamlit/ ./.streamlit/

# Expose port untuk Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Jalankan aplikasi
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
