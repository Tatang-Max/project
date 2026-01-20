import streamlit as st
import pandas as pd
import joblib
import os
import datetime
from google.cloud.exceptions import NotFound 
from google.cloud import bigquery
from google.cloud import storage # <--- Sekarang ini bakal NYALA (kepake)
from google.oauth2 import service_account

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CardioCheck",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [BAGIAN YANG HILANG TADI: KONFIGURASI] ---
# Pastikan ini ada biar kodingan tau mau nyimpen kemana
PROJECT_ID = "cloudcomputing-484913" 
DATASET_ID = "dataset_jantung"    
TABLE_ID = "history_prediksi"     
BUCKET_NAME = "bucket-jantung-data" # <--- Ini target Storage lu

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    file_path = 'rf.pkl' # Pastikan nama file sesuai repo lu
    if not os.path.exists(file_path):
        file_path = 'random_forest_medical.pkl' # Fallback
    if not os.path.exists(file_path):
        return None
    return joblib.load(file_path)

data = load_model()
if data is None:
    st.error("⚠️ File Model (.pkl) tidak ditemukan.")
    st.stop()

model = data['model']
saved_features = data['features'] 

# --- 3. SIDEBAR ---
st.sidebar.title("🎛️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Metode Input:", [" Input Manual", "📂 Upload File (Batch)"])
st.sidebar.divider()
user_input_data = {}

# --- FUNGSI AUTH ---
def get_credentials():
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        return creds
    else:
        st.error("⚠️ Secrets belum disetting!")
        return None

# --- FUNGSI SAVE KE BIGQUERY ---
def save_to_bigquery(df, project_id, dataset_id, table_id):
    try:
        creds = get_credentials()
        if not creds: return False, "Auth Gagal"

        client = bigquery.Client(credentials=creds, project=project_id)
        
        # Cek Dataset (Auto-Create)
        dataset_ref = f"{project_id}.{dataset_id}"
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            client.create_dataset(dataset)
            print(f"Dataset {dataset_id} dibuat!")
        
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        # Bersihkan nama kolom
        df_clean = df.copy()
        df_clean.columns = [c.replace(' ', '_').replace('-', '_').replace('.', '') for c in df_clean.columns]
        df_clean['waktu_prediksi'] = datetime.datetime.now()
        
        job_config = bigquery.LoadJobConfig(
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            write_disposition="WRITE_APPEND", 
        )
        job = client.load_table_from_dataframe(df_clean, table_ref, job_config=job_config)
        job.result() 
        return True, "Data masuk BigQuery."
    except Exception as e:
        return False, f"BigQuery Error: {str(e)}"

# --- [BAGIAN YANG HILANG TADI: FUNGSI UPLOAD STORAGE] ---
def upload_to_gcs(file_obj, filename, bucket_name):
    """Upload file mentah ke Google Cloud Storage"""
    try:
        creds = get_credentials()
        if not creds: return False

        # --- DISINI LIBRARY 'storage' DIPAKE ---
        client = storage.Client(credentials=creds, project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"uploads/{filename}") 
        
        # Reset pointer & Upload
        file_obj.seek(0)
        blob.upload_from_file(file_obj)
        file_obj.seek(0) # Balikin pointer biar bisa dibaca Pandas
        return True
    except Exception as e:
        print(f"GCS Error: {e}")
        return False

# --- UI UTAMA ---
st.title(" CardioCheck")
st.markdown("### Sistem Deteksi Dini Risiko Penyakit Jantung Akut")

# --- MODE 1: MANUAL ---
if mode == " Input Manual":
    st.sidebar.subheader("Masukkan Data Klinis:")
    for feature in saved_features:
        label = feature
        if feature == 'Gender':
            gender_option = st.sidebar.selectbox(f"{label}", ["Laki-laki", "Perempuan"])
            user_input_data[feature] = 1 if gender_option == "Laki-laki" else 0
            continue
        
        if feature in ['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure']:
            if 'Age' in feature: min_v, max_v, val = 20, 110, 55
            elif 'Heart rate' in feature: min_v, max_v, val = 40, 200, 80
            elif 'Systolic' in feature: min_v, max_v, val = 50, 250, 120
            elif 'Diastolic' in feature: min_v, max_v, val = 30, 150, 80
            else: min_v, max_v, val = 0, 300, 0
            user_input_data[feature] = st.sidebar.number_input(f"{label}", min_value=int(min_v), max_value=int(max_v), value=int(val), step=1)
        else:
            if 'Blood sugar' in feature: min_v, max_v, val, step, fmt = 30.0, 600.0, 120.0, 1.0, "%.1f"
            elif 'CK-MB' in feature: min_v, max_v, val, step, fmt = 0.0, 400.0, 5.0, 0.1, "%.2f"
            elif 'Troponin' in feature: min_v, max_v, val, step, fmt = 0.0, 20.0, 0.01, 0.001, "%.3f"
            else: min_v, max_v, val, step, fmt = 0.0, 1000.0, 0.0, 0.1, "%.2f"
            user_input_data[feature] = st.sidebar.number_input(f"{label}", min_value=float(min_v), max_value=float(max_v), value=float(val), step=float(step), format=fmt)

    st.info(" Mode Input Manual aktif.")
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.caption("Preview Data Input:")
        input_df = pd.DataFrame([user_input_data])
        st.dataframe(input_df.T, use_container_width=True)
    with col2:
        st.write("### Hasil Analisis")
        if st.button("🔍 ANALISIS SEKARANG", type="primary"):
            with st.spinner('Memproses...'):
                try:
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    risk_score = proba[1]
                    
                    # Simpan ke BigQuery
                    bq_df = input_df.copy()
                    bq_df['hasil_prediksi'] = int(prediction)
                    bq_df['probabilitas_risiko'] = float(risk_score)
                    success_bq, msg_bq = save_to_bigquery(bq_df, PROJECT_ID, DATASET_ID, TABLE_ID)

                    if prediction == 1:
                        st.error(f"🚨 BERISIKO TINGGI ({risk_score*100:.1f}%)")
                        st.progress(risk_score)
                    else:
                        st.success(f"✅ AMAN ({risk_score*100:.1f}%)")
                        st.progress(risk_score)
                    
                    if success_bq: st.toast("✅ Data tersimpan ke Cloud", icon="☁️")
                    else: st.toast(f"⚠️ {msg_bq}", icon="❌")
                except Exception as e: st.error(f"Error: {e}")

# --- MODE 2: UPLOAD BATCH ---
elif mode == "📂 Upload File (Batch)": 
    st.info(" Mode Batch aktif.")
    uploaded_file = st.file_uploader("Upload file data pasien", type=["xlsx", "csv"])

    with st.expander("ℹ Format File"):
        st.code(f"{', '.join(saved_features)}", language="text")

    if uploaded_file is not None:
        try:
            # --- [BAGIAN INI YANG BIKIN STORAGE NYALA] ---
            # Kita panggil fungsi upload_to_gcs disini
            with st.spinner("Backup file ke Cloud Storage..."):
                if upload_to_gcs(uploaded_file, uploaded_file.name, BUCKET_NAME):
                    st.toast("File aman di Cloud Storage!", icon="📦")
                else:
                    st.warning("Gagal backup ke Storage (Cek permission), tapi analisis tetap jalan.")

            # Baca File
            if uploaded_file.name.endswith('.csv'):
                try: df_upload = pd.read_csv(uploaded_file, sep=';')
                except: df_upload = pd.read_csv(uploaded_file, sep=',')
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            # Validasi
            missing_cols = [col for col in saved_features if col not in df_upload.columns]
            if missing_cols:
                st.error(f"Kolom hilang: {missing_cols}")
            else:
                st.success(f"File dimuat: {len(df_upload)} Baris.")
                if st.button(" JALANKAN ANALISIS BATCH", type="primary"):
                    with st.spinner('Menganalisis...'):
                        X_batch = df_upload[saved_features]
                        predictions = model.predict(X_batch)
                        probabilities = model.predict_proba(X_batch)[:, 1]
                        
                        # Simpan ke BigQuery
                        bq_df = df_upload.copy()
                        bq_df['hasil_prediksi'] = predictions
                        bq_df['probabilitas_risiko'] = probabilities
                        success_bq, msg_bq = save_to_bigquery(bq_df, PROJECT_ID, DATASET_ID, TABLE_ID)
                        
                        # Tampilan
                        result_df = df_upload.copy()
                        result_df['Risiko (%)'] = probabilities
                        result_df['Status'] = ["🔴 BAHAYA" if p == 1 else "🟢 AMAN" for p in predictions]
                        cols = ['Status', 'Risiko (%)'] + [c for c in result_df.columns if c not in ['Status', 'Risiko (%)']]
                        st.dataframe(result_df[cols], use_container_width=True)
                        
                        high_risk_count = list(predictions).count(1)
                        st.metric("Pasien Berisiko", f"{high_risk_count}", delta="Perhatian")
                        
                        if success_bq: st.caption("✅ Masuk BigQuery")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Powered by Streamlit & GCP")