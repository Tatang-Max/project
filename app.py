import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import gspread 
import uuid # Nambah ini buat generate nama file unik
from oauth2client.service_account import ServiceAccountCredentials
from supabase import create_client, Client # Nambah ini buat Supabase

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CardioCheck",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONFIGURASI DATABASE & STORAGE ---
SHEET_NAME = "Database_Jantung" 
BUCKET_NAME = "dokumen-pasien" # Pastikan bucket ini udah dibuat di Supabase & set Public

# Fungsi Inisialisasi Supabase
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_supabase()

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    # Sesuai file pkl lu le
    file_path = 'random_forest_medical.pkl' 
    
    if not os.path.exists(file_path):
        return None
        
    return joblib.load(file_path)

data = load_model()
if data is None:
    st.error("⚠️ File Model 'random_forest_medical.pkl' tidak ditemukan di folder!")
    st.stop()

model = data['model']
saved_features = data['features'] 

# --- 3. FUNGSI KONEKSI KE SHEETS & STORAGE ---
def get_sheet_connection():
    if "gcp_service_account" not in st.secrets:
        return None
    
    creds_dict = st.secrets["gcp_service_account"]
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

# Fungsi Upload ke Supabase
def upload_evidence(file_obj):
    if not supabase: return None
    try:
        file_bytes = file_obj.getvalue()
        file_ext = file_obj.name.split('.')[-1]
        file_name = f"evidence_{uuid.uuid4()}.{file_ext}"
        
        # Upload ke Supabase
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": file_obj.type}
        )
        
        # Ambil Public URL
        return supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
    except Exception as e:
        st.error(f"Gagal upload storage: {e}")
        return None

def save_to_database_awan(df):
    try:
        client = get_sheet_connection()
        if not client: return False, "Secrets belum disetting!"

        try:
            sheet = client.open(SHEET_NAME).sheet1
        except:
            return False, f"Sheet '{SHEET_NAME}' gak ketemu."

        df_save = df.copy()
        df_save.columns = [c.replace(' ', '_').replace('-', '_').replace('.', '') for c in df_save.columns]
        df_save['waktu_prediksi'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sheet.append_rows(df_save.values.tolist())
        return True, "Data tersimpan di Database Awan."
    except Exception as e:
        return False, str(e)

# --- 4. SIDEBAR & ALUR KERJA SISTEM ---
st.sidebar.title("🎛️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Metode Input:", [" Input Manual", "📂 Upload File (Batch)"])
st.sidebar.divider()
user_input_data = {}

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
    
    # --- TAMBAHAN: Upload Bukti Pendukung ---
    st.markdown("#### 📁 Dokumen Pendukung (Opsional)")
    evidence_file = st.file_uploader("Upload hasil lab/rontgen (JPG/PNG/PDF)", type=['jpg', 'jpeg', 'png', 'pdf'])
    
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
                    # 1. Prediksi
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    risk_score = proba[1]
                    
                    bq_df = input_df.copy()
                    bq_df['hasil_prediksi'] = int(prediction)
                    bq_df['probabilitas_risiko'] = float(risk_score)
                    
                    # 2. Upload ke Supabase (Jika ada file)
                    file_url = "-"
                    if evidence_file is not None:
                         with st.spinner('Mengupload dokumen ke Cloud Storage...'):
                            url_result = upload_evidence(evidence_file)
                            if url_result:
                                file_url = url_result
                    
                    # Masukkan link ke dataframe
                    bq_df['link_bukti'] = file_url

                    # 3. Simpan ke Google Sheets
                    success_bq, msg_bq = save_to_database_awan(bq_df)

                    if prediction == 1:
                        st.error(f"🚨 BERISIKO TINGGI ({risk_score*100:.1f}%)")
                        st.progress(risk_score)
                    else:
                        st.success(f"✅ AMAN ({risk_score*100:.1f}%)")
                        st.progress(risk_score)
                        
                    if success_bq: st.toast("✅ Data & Dokumen tersimpan", icon="☁️")
                    else: st.toast(f"⚠️ {msg_bq}", icon="❌")
                    
                    # Tampilkan link jika berhasil upload
                    if file_url != "-":
                        st.caption(f"🔗 Dokumen tersimpan: {file_url}")

                except Exception as e: st.error(f"Error: {e}")

# --- MODE 2: UPLOAD BATCH ---
elif mode == "📂 Upload File (Batch)": 
    st.info(" Mode Batch aktif.")
    uploaded_file = st.file_uploader("Upload file data pasien", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try: df_upload = pd.read_csv(uploaded_file, sep=';')
                except: df_upload = pd.read_csv(uploaded_file, sep=',')
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            missing_cols = [col for col in saved_features if col not in df_upload.columns]
            if missing_cols:
                st.error(f"Kolom hilang: {missing_cols}")
            else:
                st.success(f"File dimuat: {len(df_upload)} Baris.")
                if st.button(" JALANKAN ANALISIS BATCH", type="primary"):
                    with st.spinner('Menganalisis & Menyimpan...'):
                        X_batch = df_upload[saved_features]
                        predictions = model.predict(X_batch)
                        probabilities = model.predict_proba(X_batch)[:, 1]
                        
                        bq_df = df_upload.copy()
                        bq_df['hasil_prediksi'] = predictions
                        bq_df['probabilitas_risiko'] = probabilities
                        bq_df['link_bukti'] = "-" # Batch default strip dulu biar kolom konsisten
                        
                        success_bq, msg_bq = save_to_database_awan(bq_df)
                        
                        st.dataframe(bq_df, use_container_width=True)
                        if success_bq: st.success("✅ Data Batch disimpan ke Cloud Database!")
                        else: st.error(f"Gagal simpan: {msg_bq}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Powered by Streamlit, Supabase Storage & Google Sheets")