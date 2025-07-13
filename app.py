import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model (.keras, pastikan file tersedia di folder yang sama atau path yang benar)
model = load_model("model_bandeng_baru.keras", compile=False)

# Label mapping: {'segar': 0, 'tidak segar': 1}
label_map = {0: "Segar", 1: "Tidak Segar"}

# Judul aplikasi
st.title("Deteksi Kesegaran Ikan Bandeng 🐟")
st.write("Upload foto ikan untuk mengetahui apakah masih segar atau tidak.")

# Upload gambar
uploaded_file = st.file_uploader("📤 Pilih gambar ikan bandeng", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_column_width=True)

        # Preprocessing
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        pred = model.predict(img_array)[0][0]
        kelas = label_map[0] if pred < 0.5 else label_map[1]

        # Tampilkan hasil
        st.markdown(f"### ✅ Hasil: **{kelas.upper()}** ({pred:.2f})")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi. Error: {str(e)}")
