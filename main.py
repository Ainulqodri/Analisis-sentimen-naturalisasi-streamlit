# streamlit_sentiment_app.py
import streamlit as st
import os
import requests
import io
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import preprocessing
from preprocessing import full_preprocess, remove_stopwords

# === Fungsi prediksi ===
def predict_sentiment(texts, model, tokenizer, max_len=50):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    predictions = model.predict(padded)
    labels = (predictions > 0.5).astype(int).flatten()
    confidences = predictions.flatten()
    return labels, confidences

# === Sidebar Navigasi ===
menu = st.sidebar.radio("📚 Pilih Menu", ["Home", "Analisis", "Wordcloud", "Tentang Model"])

# === Load model dan tokenizer hanya sekali ===
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url}...")
        r = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(r.content)

@st.cache_resource
def load_model_tokenizer():
    os.makedirs("model", exist_ok=True)
    os.makedirs("app", exist_ok=True)

    # URLs dari GitHub Releases
    model_url = "https://github.com/Ainulqodri/Analisis-sentimen-naturalisasi-streamlit/releases/download/model88/model_lstm_88.keras"
    tokenizer_url = "https://github.com/Ainulqodri/Analisis-sentimen-naturalisasi-streamlit/releases/download/model88/tokenizer_88.pkl"
    label_url = "https://github.com/Ainulqodri/Analisis-sentimen-naturalisasi-streamlit/releases/download/model88/label_encoder_88.pkl"

    # Path lokal setelah download
    model_path = "model/model_lstm_88.keras"
    tokenizer_path = "app/tokenizer_88.pkl"
    label_path = "app/label_encoder_88.pkl"

    # Download jika belum ada
    download_file(model_url, model_path)
    download_file(tokenizer_url, tokenizer_path)
    download_file(label_url, label_path)

    # Load model dan tokenizer
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_tokenizer()

# === Home Page ===
if menu == "Home":
    st.markdown("""
    <h3 style='margin-bottom: 0.5rem;'>📊 Analisis Sentimen Masyarakat terhadap Program Naturalisasi PSSI di Media Sosial X Menggunakan Deep Learning LSTM</h3>
    <p style='margin: 0.25rem 0;'><b>Nama</b>&nbsp;&nbsp;&nbsp;&nbsp;: Ainul Qodri</p>
    <p style='margin: 0.25rem 0;'><b>NIM</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 2121400049</p>
    """, unsafe_allow_html=True)

# === Analisis Page ===
elif menu == "Analisis":
    st.title("📊 Sentiment Analysis dari File CSV")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        st.session_state["uploaded_file_content"] = content
        st.session_state["analyzed"] = False

    content = st.session_state.get("uploaded_file_content", None)

    df = None
    if "clean_df" in st.session_state:
        df = st.session_state["clean_df"]
        text_col = st.session_state.get("text_col", "")
        label_col = st.session_state.get("label_col", "")

    if content is not None and not st.session_state.get("analyzed", False):
        df = pd.read_csv(io.BytesIO(content))
        df = preprocessing.filter_accounts(df, username_column="username")
        st.write("📄 Data dimuat:")
        st.dataframe(df)

        text_col = st.selectbox("Pilih kolom teks untuk dianalisis:", df.columns)
        label_col = st.selectbox("Pilih kolom label (opsional):", ["(Tidak ada)"] + list(df.columns))
        preprocess_option = st.radio("Apakah data perlu dipreprocessing?", ("Ya", "Tidak"))

        with st.spinner("🔍 Memproses dan menganalisis data..."):
            df["clean_text"] = df[text_col].astype(str).apply(full_preprocess if preprocess_option == "Ya" else str)
            # df = df.drop_duplicates(subset=["clean_text"], keep="first")
            predictions, confidences = predict_sentiment(df["clean_text"], model, tokenizer)
            df["predicted_label"] = predictions
            df["confidence"] = confidences
            df["sentimen_prediksi"] = label_encoder.inverse_transform(df["predicted_label"])

            st.session_state["clean_df"] = df
            st.session_state["text_col"] = text_col
            st.session_state["label_col"] = label_col
            st.session_state["analyzed"] = True

    if df is not None:
        st.success("✅ Data sudah dianalisis!")
        st.dataframe(df[["clean_text", "sentimen_prediksi", "confidence"]])

        # Fungsi untuk menampilkan jumlah dan persentase
        def format_label(pct, allvals):
            total = sum(allvals)
            count = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({count})"

        # === Pie Chart Distribusi ===
        st.subheader("📊 Distribusi Sentimen (Pie Chart)")
        # # Tampilkan jumlah total data
        # st.write("Jumlah data total:", len(df))

        # # Tampilkan jumlah yang berhasil diprediksi
        # st.write("Jumlah data dengan label prediksi:", df["predicted_label"].notnull().sum())

        # # Tampilkan jumlah yang gagal (NaN)
        # st.write("Jumlah data tanpa prediksi:", df["predicted_label"].isnull().sum())
        pie_data = df["predicted_label"].value_counts().rename({0: "Negatif", 1: "Positif"})
        fig1, ax1 = plt.subplots()
        ax1.pie(
            pie_data,
            labels=pie_data.index,
            autopct=lambda pct: format_label(pct, pie_data),
            startangle=90
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # === Visualisasi Sentimen Berdasarkan Waktu ===
        if "created_at" in df.columns:
            st.subheader("📈 Distribusi Sentimen Bulanan (Line Chart)")

            try:
                # Konversi created_at ke datetime
                df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
                df = df.dropna(subset=["created_at"])

                # Ambil bulan saja
                df["created_week"] = df["created_at"].dt.to_period("W").dt.start_time

                # Hitung jumlah tweet per bulan dan label
                timeline_monthly = df.groupby(["created_week", "predicted_label"]).size().unstack(fill_value=0)
                timeline_monthly.columns = ["Negatif" if c == 0 else "Positif" for c in timeline_monthly.columns]

                # Plot line chart bulanan
                fig_time, ax_time = plt.subplots(figsize=(10, 5))
                timeline_monthly.plot(kind="line", ax=ax_time, marker="o")

                ax_time.set_title("Distribusi Sentimen Bulanan")
                ax_time.set_xlabel("Bulan")
                ax_time.set_ylabel("Jumlah Tweet")
                ax_time.legend(title="Sentimen")

                ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig_time.autofmt_xdate(rotation=45)

                st.pyplot(fig_time)

            except Exception as e:
                st.warning(f"⚠️ Gagal memproses waktu: {e}")



        # Confusion Matrix dan Classification Report
        if label_col != "(Tidak ada)":
            st.subheader("🧮 Confusion Matrix & Classification Report")
            df_clean = df.dropna(subset=[label_col])
            df_clean[label_col] = df_clean[label_col].astype(str).str.strip().str.capitalize()
            df_clean = df_clean[df_clean[label_col].isin(["Positif", "Negatif"])]

            y_true = label_encoder.transform(df_clean[label_col])
            y_pred = df_clean["predicted_label"]

            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)

            fig2, ax2 = plt.subplots(figsize=(3, 2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,cbar=False, annot_kws={"size": 6})
            ax2.set_xlabel("Predicted", fontsize=6)
            ax2.set_ylabel("Actual", fontsize=6)
            ax2.tick_params(axis='both', labelsize=6)
            st.pyplot(fig2)

            st.text("Classification Report:")
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(report_df)

            # Bar chart perbandingan
            st.subheader("📊 Perbandingan Label Aktual dan Prediksi")
            actual_counts = df_clean[label_col].value_counts().sort_index()
            predicted_counts = df_clean["sentimen_prediksi"].value_counts().sort_index()
            labels = ["Negatif", "Positif"]
            actual_values = [actual_counts.get(label, 0) for label in labels]
            predicted_values = [predicted_counts.get(label, 0) for label in labels]
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots()
            bars1 = ax.bar(x - width/2, actual_values, width, label='Aktual', color='red')
            bars2 = ax.bar(x + width/2, predicted_values, width, label='Prediksi', color='blue')
            ax.set_ylabel('Jumlah')
            ax.set_title('Perbandingan Label Aktual dan Prediksi')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(title="Keterangan")
            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            st.pyplot(fig)

# === Wordcloud Page ===
elif menu == "Wordcloud":
    st.title("☁️ Wordcloud Visualisasi Teks")
    if "clean_df" in st.session_state and "clean_text" in st.session_state["clean_df"]:
        clean_texts = st.session_state["clean_df"]["clean_text"].astype(str).apply(remove_stopwords)
        text_all = " ".join(clean_texts)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Bar kata
        tokens_cleaned = clean_texts.str.cat(sep=' ').split()
        word_freq = Counter(tokens_cleaned)
        common_words = word_freq.most_common(10)
        words, counts = zip(*common_words)
        st.subheader("📊 Kata Paling Sering Muncul")
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=list(counts), y=list(words), orient='h', ax=ax_bar, palette="viridis")
        ax_bar.set_xlabel("Jumlah Kemunculan")
        ax_bar.set_ylabel("Kata")
        st.pyplot(fig_bar)
    else:
        st.warning("⚠️ Silakan lakukan analisis terlebih dahulu di menu 'Analisis'.")

# === Tentang Model Page ===
elif menu == "Tentang Model":
    st.title("🧠 Tentang Model LSTM")
    st.markdown("""
    - **Model:** LSTM
    - **Keluaran:** 2 kelas (biner) dengan aktivasi `sigmoid`
    - **Maxlen:** 50
    - **Dataset:** Hasil preprocessing media sosial X
    - **Evaluasi:** Confusion matrix, classification report (`precision`,`recall`,`f1-score`)
    """)
