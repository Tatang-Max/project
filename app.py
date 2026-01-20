import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CardioCheck ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    file_path = 'random_forest_medical.pkl'
    if not os.path.exists(file_path):
        return None
    return joblib.load(file_path)

# Load data model
data = load_model()

if data is None:
    st.error("⚠️ File 'random_forest_medical.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

model = data['model']
# Mengambil list fitur yang dipake pas training (biar urutannya bener)
saved_features = data['features'] 

# --- 3. SIDEBAR: PILIH MODE ---
st.sidebar.title("🎛️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Metode Input:", [" Input Manual", "📂 Upload File (Batch)"])

st.sidebar.divider()

# Variabel buat nyimpen data inputan manual
user_input_data = {}

# --- MODE 1: INPUT MANUAL (Final Update: Gender Dropdown) ---
if mode == " Input Manual":
    st.sidebar.subheader("Masukkan Data Klinis:")
    
    for feature in saved_features:
        label = feature
        
        # 1. KHUSUS GENDER (Dropdown Menu)
        if feature == 'Gender':
            # Tampilan di layar: Teks
            gender_option = st.sidebar.selectbox(
                f"{label}", 
                ["Laki-laki", "Perempuan"],
                help="Jenis Kelamin Pasien"
            )
            
            # Logic Backend: Ubah jadi Angka (1 = Laki, 0 = Perempuan)
            # Sesuaikan kalau model lu kebalik (misal 0=Laki), tinggal tuker angkanya
            user_input_data[feature] = 1 if gender_option == "Laki-laki" else 0
            
            continue # Lanjut ke fitur berikutnya, jangan bikin input angka lagi buat gender
        
        # 2. INPUT DATA INTEGER (Bilangan Bulat)
        if feature in ['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure']:
            
            if 'Age' in feature:
                min_v, max_v, val = 20, 110, 55
                help_txt = "Usia Pasien (Tahun)"
            elif 'Heart rate' in feature:
                min_v, max_v, val = 40, 200, 80
                help_txt = "Detak Jantung (bpm)"
            elif 'Systolic' in feature:
                min_v, max_v, val = 50, 250, 120
                help_txt = "Tekanan Darah Sistolik/Atas (mmHg)"
            elif 'Diastolic' in feature:
                min_v, max_v, val = 30, 150, 80
                help_txt = "Tekanan Darah Diastolik/Bawah (mmHg)"
            else:
                min_v, max_v, val = 0, 300, 0
                help_txt = ""

            user_input_data[feature] = st.sidebar.number_input(
                f"{label}", 
                min_value=int(min_v), 
                max_value=int(max_v), 
                value=int(val), 
                step=1, 
                help=help_txt
            )
            
        # 3. INPUT DATA FLOAT (Bilangan Koma)
        else:
            if 'Blood sugar' in feature:
                min_v, max_v, val, step, fmt = 30.0, 600.0, 120.0, 1.0, "%.1f"
                help_txt = "Gula Darah Sewaktu (mg/dL)"
            elif 'CK-MB' in feature:
                min_v, max_v, val, step, fmt = 0.0, 400.0, 5.0, 0.1, "%.2f"
                help_txt = "Kadar Enzim CK-MB (unit/L)"
            elif 'Troponin' in feature:
                min_v, max_v, val, step, fmt = 0.0, 20.0, 0.01, 0.001, "%.3f"
                help_txt = "Kadar Troponin (ng/mL)"
            else:
                min_v, max_v, val, step, fmt = 0.0, 1000.0, 0.0, 0.1, "%.2f"
                help_txt = ""
                
            user_input_data[feature] = st.sidebar.number_input(
                f"{label}", 
                min_value=float(min_v), 
                max_value=float(max_v), 
                value=float(val), 
                step=float(step), 
                format=fmt, 
                help=help_txt
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
            # Baca File
            if uploaded_file.name.endswith('.csv'):
                # Coba baca pake separator ; dulu, kalau error coba ,
                try:
                    df_upload = pd.read_csv(uploaded_file, sep=';')
                    if len(df_upload.columns) < 2: # Kalau gagal parsing (kolom jadi 1)
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
                        probabilities = model.predict_proba(X_batch)[:, 1] # Ambil probabilitas kelas 1
                        
                        # Bikin DataFrame Hasil
                        result_df = df_upload.copy()
                        
                        # Tambah kolom hasil
                        result_df['Risiko (%)'] = probabilities
                        result_df['Status'] = ["🔴 BERISIKO" if p == 1 else "🟢 AMAN" for p in predictions]
                        
                        # Pindahkan kolom hasil ke depan biar gampang dilihat
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