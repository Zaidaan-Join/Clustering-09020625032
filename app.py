import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sistem Segmentasi Pelanggan RFM",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler_rfm.pkl')
    model = joblib.load('model_kmeans_rfm.pkl')
    return scaler, model

try:
    scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat file model (.pkl): {e}")
    st.info("Pastikan file 'scaler_rfm.pkl' dan 'model_kmeans_rfm.pkl' diunggah melalui menu 'Add file -> Upload files' di GitHub.")
    st.stop()

# Sesuaikan nomor klaster (0, 1, 2, 3) dengan hasil rata-rata centroid di notebook Anda
SEGMENT_INFO = {
    0: {
        "nama": "Champions (Pelanggan Terbaik)",
        "deskripsi": "Baru saja bertransaksi, sangat sering membeli, dan memiliki nilai belanja tertinggi.",
        "strategi": "Berikan akses eksklusif produk baru, program loyalitas VIP, dan tawarkan program referral."
    },
    1: {
        "nama": "Loyal / Potential Loyalists (Pelanggan Potensial)",
        "deskripsi": "Cukup sering bertransaksi dengan nilai belanja menengah ke atas.",
        "strategi": "Lakukan upselling/cross-selling produk terkait dan berikan poin reward untuk meningkatkan frekuensi."
    },
    2: {
        "nama": "At Risk (Berisiko Beralih)",
        "deskripsi": "Dulu sering berbelanja dengan nilai tinggi, tetapi sudah cukup lama tidak kembali.",
        "strategi": "Kirimkan kampanye win-back melalui email personal, diskon khusus, atau survei kepuasan."
    },
    3: {
        "nama": "Hibernating / Lost (Pelanggan Pasif)",
        "deskripsi": "Sudah sangat lama tidak bertransaksi dengan frekuensi dan nilai belanja rendah.",
        "strategi": "Jangan alokasikan biaya pemasaran terlalu besar; cukup sertakan dalam promo diskon musiman massal."
    }
}

st.title("📊 Dashboard Segmentasi Pelanggan (RFM)")
st.markdown("Aplikasi prediksi segmen pelanggan berbasis **K-Means Clustering** untuk mendukung strategi pemasaran.")

tab1, tab2 = st.tabs(["🔍 Prediksi Pelanggan Tunggal", "📂 Segmentasi Massal (Upload CSV)"])

with tab1:
    st.subheader("Masukkan Data RFM Pelanggan")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (Hari sejak transaksi terakhir)", min_value=0, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (Total jumlah transaksi)", min_value=1, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (Total nilai belanja)", min_value=0.0, value=1500.0, step=50.0)

    if st.button("Analisis Segmen Pelanggan", type="primary"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        
        cluster = int(model.predict(input_scaled)[0])
        info = SEGMENT_INFO.get(cluster, {
            "nama": f"Klaster {cluster}",
            "deskripsi": "Profil klaster belum didefinisikan.",
            "strategi": "Lakukan evaluasi lebih lanjut pada karakteristik klaster ini."
        })
        
        st.divider()
        st.success(f"**Hasil Prediksi: Klaster {cluster} — {info['nama']}**")
        st.write(f"**Karakteristik:** {info['deskripsi']}")
        st.info(f"**Rekomendasi Tindakan:** {info['strategi']}")

with tab2:
    st.subheader("Unggah Dataset Pelanggan")
    st.markdown("Pastikan file CSV memiliki kolom: `Recency`, `Frequency`, dan `Monetary`.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = {'Recency', 'Frequency', 'Monetary'}
        
        if not required_cols.issubset(df.columns):
            st.error(f"File CSV harus mengandung kolom: {', '.join(required_cols)}")
        else:
            features = df[['Recency', 'Frequency', 'Monetary']]
            features_scaled = scaler.transform(features)
            df['Cluster'] = model.predict(features_scaled)
            df['Segmen'] = df['Cluster'].map(lambda x: SEGMENT_INFO.get(x, {}).get('nama', f'Klaster {x}'))
            
            st.write("### Ringkasan Distribusi Segmen")
            st.bar_chart(df['Segmen'].value_counts())
            
            st.write("### Pratinjau Hasil Segmentasi")
            st.dataframe(df.head(20))
            
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh Hasil Segmentasi (CSV)",
                data=csv_output,
                file_name="hasil_segmentasi_rfm.csv",
                mime="text/csv"
            )
