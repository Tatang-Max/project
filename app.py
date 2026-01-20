import streamlit as st
import pandas as pd
import joblib
import os
import datetime
# [BARU] Import Library Google Cloud
from google.cloud import bigquery
from google.cloud import storage

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CardioCheck AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [BARU] KONFIGURASI GCP (WAJIB DIISI) ---
# Ganti dengan ID Project, Nama Dataset, dan Nama Bucket kamu
PROJECT_ID = "bigdataea-484802" 
DATASET_ID = "dataset_jantung"    # Pastikan dataset ini udah dibuat di BigQuery
TABLE_ID = "history_prediksi"     # Nama tabel (akan dibuat otomatis jika belum ada)
BUCKET_NAME = "bucket-jantung-data" # Pastikan bucket ini udah dibuat di Cloud Storage

# --- [BARU] FUNGSI HELPER KE CLOUD ---
def save_to_bigquery(df, project_id, dataset_id, table_id):
    """Menyimpan DataFrame ke Google BigQuery"""
    try:
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        # Tambah timestamp agar tau kapan data masuk
        df['waktu_prediksi'] = datetime.datetime.now()
        
        # Load data
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND", # Tambah data baru di bawah (tidak menimpa)
        )
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result() # Tunggu sampai selesai
        return True, "Data berhasil direkam ke BigQuery."
    except Exception as e:
        return False, f"Gagal menyimpan ke BigQuery: {str(e)}"

def upload_to_gcs(file_obj, filename, bucket_name):
    """Upload file mentah ke Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"uploads/{filename}") # Masuk ke folder uploads/
        
        # Reset pointer file ke awal sebelum upload
        file_obj.seek(0)
        blob.upload_from_file(file_obj)
        # Reset lagi pointer biar bisa dibaca pandas setelah ini
        file_obj.seek(0)
        return True
    except Exception as e:
        return False

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    file_path = 'rf.pkl' # Pastikan nama file model sesuai (tadi kamu sebut rf.pkl)
    # Fallback nama file lama jika rf.pkl tidak ada
    if not os.path.exists(file_path):
        file_path = 'random_forest_medical.pkl'
        
    if not os.path.exists(file_path):
        return None
    return joblib.load(file_path)

# Load data model
data = load_model()

if data is None:
    st.error("⚠️ File Model (.pkl) tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# Handle jika model disimpan dictionary atau langsung object model
if isinstance(data, dict):
    model = data['model']
    saved_features = data['features']
else:
    model = data
    # [PENTING] Jika features tidak tersimpan di dict, kita harus define manual 
    # atau ambil dari logic sebelumnya. Asumsi di sini data berupa dict sesuai kodemu.
    st.error("Format model tidak sesuai struktur dictionary {model, features}.")
    st.stop()

# --- 3. SIDEBAR: PILIH MODE ---
st.sidebar.title("🎛️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Metode Input:", [" Input Manual", "📂 Upload File (Batch)"])

st.sidebar.divider()

# Variabel buat nyimpen data inputan manual
user_input_data = {}

# --- MODE 1: INPUT MANUAL ---
if mode == " Input Manual":
    st.sidebar.subheader("Masukkan Data Klinis:")
    
    for feature in saved_features:
        label = feature
        
        # Logic input khusus Age (Integer)
        if 'Age' in feature:
            user_input_data[feature] = st.sidebar.number_input(
                f"{label}", min_value=0, max_value=120, value=45, step=1,
                help="Usia pasien (tahun)"
            )
        # Logic input Float
        else:
            if 'Troponin' in feature:
                min_v, max_v, val, step = 0.0, 100.0, 0.01, 0.001
                help_text = "Kadar Troponin (ng/mL)"
            elif 'CK-MB' in feature:
                min_v, max_v, val, step = 0.0, 300.0, 2.0, 0.1
                help_text = "Kadar CK-MB (unit/L)"
            else:
                min_v, max_v, val, step = 0.0, 1000.0, 0.0, 1.0
                help_text = ""

            user_input_data[feature] = st.sidebar.number_input(
                f"{label}", min_value=float(min_v), max_value=float(max_v), 
                value=float(val), step=float(step), format="%.3f", help=help_text
            )

# --- 4. UI UTAMA ---
st.title(" CardioCheck AI")
st.markdown("### Sistem Deteksi Dini Risiko Penyakit Jantung Akut")

# --- LOGIC MODE 1: TAMPILAN MANUAL ---
if mode == " Input Manual":
    st.info(" Mode Input Manual aktif. Silakan isi data di sidebar sebelah kiri.")
    
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.caption("Preview Data Input:")
        input_df = pd.DataFrame([user_input_data])
        st.dataframe(input_df.T, use_container_width=True)

    with col2:
        st.write("### Hasil Analisis")
        if st.button("🔍 ANALISIS SEKARANG", type="primary", use_container_width=True):
            with st.spinner('Sedang membedah jantung virtual...'):
                try:
                    # Prediksi
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    risk_score = proba[1] # Ambil probabilitas kelas 1 (Sakit)

                    # [BARU] Simpan hasil ke BigQuery
                    input_df['hasil_prediksi'] = int(prediction)
                    input_df['probabilitas_risiko'] = float(risk_score)
                    
                    # Eksekusi simpan background
                    success_bq, msg_bq = save_to_bigquery(input_df, PROJECT_ID, DATASET_ID, TABLE_ID)

                    if prediction == 1:
                        st.error(f"🚨 **TERINDIKASI BERISIKO TINGGI**")
                        st.metric("Probabilitas Risiko", f"{risk_score*100:.1f}%", delta="Bahaya")
                        st.progress(risk_score)
                        st.warning("⚠️ **Saran:** Segera rujuk ke spesialis jantung.")
                    else:
                        st.success(f"✅ **KONDISI AMAN / NEGATIF**")
                        st.metric("Probabilitas Risiko", f"{risk_score*100:.1f}%", delta="-Aman", delta_color="normal")
                        st.progress(risk_score)
                        st.info("👍 **Saran:** Pertahankan gaya hidup sehat.")
                    
                    # Feedback Cloud
                    if success_bq:
                        st.toast("✅ Data berhasil disimpan ke Database Cloud.", icon="☁️")
                    else:
                        st.toast(f"⚠️ {msg_bq}", icon="❌")
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# --- LOGIC MODE 2: TAMPILAN UPLOAD BATCH ---
elif mode == " Upload File (Batch)":
    st.info(" Mode Batch aktif. Upload file Excel/CSV berisi data pasien banyak sekaligus.")
    
    # Widget Upload
    uploaded_file = st.file_uploader("Upload file data pasien", type=["xlsx", "csv"])

    # Template Download Helper
    with st.expander("ℹ Format File yang Dibutuhkan"):
        st.write("Pastikan file memiliki kolom berikut (case-sensitive):")
        st.code(f"{', '.join(saved_features)}", language="text")
        st.caption("Kolom 'Result' atau target tidak diperlukan.")

    if uploaded_file is not None:
        try:
            # [BARU] Upload file mentah ke Cloud Storage (GCS)
            with st.spinner("Mengamankan file ke Cloud Storage..."):
                upload_to_gcs(uploaded_file, uploaded_file.name, BUCKET_NAME)
            
            st.toast("File berhasil di-backup ke Cloud Storage!", icon="📦")

            # Baca File
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file, sep=';')
                    if len(df_upload.columns) < 2: 
                        uploaded_file.seek(0)
                        df_upload = pd.read_csv(uploaded_file, sep=',')
                except:
                    df_upload = pd.read_csv(uploaded_file, sep=',')
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            # Validasi Kolom
            missing_cols = [col for col in saved_features if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f" File tidak valid! Kolom berikut hilang: {missing_cols}")
            else:
                st.success(f" File berhasil dimuat! Total Data: {len(df_upload)} Baris.")
                
                if st.button(" JALANKAN ANALISIS BATCH", type="primary"):
                    with st.spinner('Menganalisis semua pasien...'):
                        
                        # Ambil hanya kolom yang dibutuhkan model
                        X_batch = df_upload[saved_features]
                        
                        # Prediksi Massal
                        predictions = model.predict(X_batch)
                        probabilities = model.predict_proba(X_batch)[:, 1]
                        
                        # Bikin DataFrame Hasil
                        result_df = df_upload.copy()
                        result_df['Risiko (%)'] = probabilities
                        result_df['Status'] = ["🔴 BERISIKO" if p == 1 else "🟢 AMAN" for p in predictions]

                        # [BARU] Simpan hasil analisa massal ke BigQuery
                        bq_df = df_upload.copy()
                        bq_df['hasil_prediksi'] = predictions
                        bq_df['probabilitas_risiko'] = probabilities
                        
                        success_bq, msg_bq = save_to_bigquery(bq_df, PROJECT_ID, DATASET_ID, TABLE_ID)
                        if success_bq:
                            st.caption("✅ Hasil analisis telah disinkronisasi ke BigQuery.")
                        
                        # Pindahkan kolom hasil ke depan
                        cols = ['Status', 'Risiko (%)'] + [c for c in result_df.columns if c not in ['Status', 'Risiko (%)']]
                        result_df = result_df[cols]

                        st.write("###  Hasil Analisis Batch")
                        
                        # TAMPILKAN TABEL KEREN
                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            column_config={
                                "Risiko (%)": st.column_config.ProgressColumn(
                                    "Tingkat Bahaya",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=1,
                                ),
                                "Status": st.column_config.TextColumn(
                                    "Status Diagnosa",
                                )
                            },
                            hide_index=True
                        )
                        
                        # Statistik Ringkas
                        high_risk_count = list(predictions).count(1)
                        st.metric("Total Pasien Berisiko Tinggi", f"{high_risk_count} Orang", delta="Perhatian")

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

st.divider()
st.caption("Dibuat dengan ☕ dan Python.")