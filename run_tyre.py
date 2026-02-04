import cv2
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog

# --- KONFIGURASI ---
MODEL_PATH = 'model_ban.h5' # Pastikan nama file model benar
IMG_SIZE = (150, 150)       # Harus sama dengan saat training

# Cek Model
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File {MODEL_PATH} tidak ditemukan!")
    exit()

print("Sedang memuat sistem AI... Mohon tunggu.")
model = tf.keras.models.load_model(MODEL_PATH)

# Cek apakah model minta input Grayscale (1 channel) atau RGB (3 channel)
input_shape = model.input_shape
EXPECTED_CHANNELS = input_shape[-1] # Ambil angka terakhir (1 atau 3)
print(f"Model terdeteksi menggunakan {EXPECTED_CHANNELS} channel warna.")

def proses_gambar():
    # 1. Buka File Dialog (Jendela pilih file)
    root = tk.Tk()
    root.withdraw() # Sembunyikan window utama tkinter
    
    file_path = filedialog.askopenfilename(
        title="Pilih Foto Ban untuk Dianalisis",
        initialdir=os.getcwd(), # Mulai dari folder project
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("Batal memilih file.")
        return False # User cancel

    # 2. Baca Gambar
    img_original = cv2.imread(file_path)
    if img_original is None:
        print("File gambar rusak atau tidak terbaca.")
        return True

    # 3. Preprocessing (Menyesuaikan dengan Model)
    img_ai = cv2.resize(img_original, IMG_SIZE)
    
    if EXPECTED_CHANNELS == 1:
        # Jika model Grayscale, ubah gambar jadi hitam putih
        img_ai = cv2.cvtColor(img_ai, cv2.COLOR_BGR2GRAY)
        img_array = tf.keras.preprocessing.image.img_to_array(img_ai)
    else:
        # Jika model RGB
        img_array = tf.keras.preprocessing.image.img_to_array(img_ai)
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Normalisasi
    
    # 4. Prediksi
    prediction = model.predict(img_array, verbose=0)
    
    # Cek apakah output model binary (1 neuron) atau categorical (3 neuron)
    output_neurons = model.output_shape[-1]
    
    label = ""
    color = (0,0,0)
    confidence = 0.0
    
    if output_neurons == 1:
        # LOGIKA BINARY (Good=1, Defective=0)
        score = prediction[0][0]
        if score > 0.5:
            confidence = score * 100
            label = f"LAYAK JALAN ({confidence:.1f}%)"
            color = (0, 200, 0) # Hijau
        else:
            confidence = (1 - score) * 100
            label = f"BAHAYA! BOTAK ({confidence:.1f}%)"
            color = (0, 0, 200) # Merah
            
    else:
        # LOGIKA CATEGORICAL (3 Kelas: Defect, Good, Negative)
        # Asumsi urutan: 0=Rusak, 1=Bagus, 2=Bukan Ban
        idx = np.argmax(prediction[0])
        conf_val = np.max(prediction[0])
        
        if idx == 2:
            label = "BUKAN BAN"
            color = (100, 100, 100) # Abu-abu
        elif idx == 1:
            label = f"LAYAK JALAN ({conf_val*100:.1f}%)"
            color = (0, 200, 0)
        else:
            label = f"BAHAYA! BOTAK ({conf_val*100:.1f}%)"
            color = (0, 0, 200)

    # 5. Tampilan Hasil (UI)
    # Resize gambar asli agar pas di layar laptop (max tinggi 600px)
    h, w, _ = img_original.shape
    aspect_ratio = w / h
    new_h = 600
    new_w = int(new_h * aspect_ratio)
    display_img = cv2.resize(img_original, (new_w, new_h))
    
    # Tambahkan Header Warna
    cv2.rectangle(display_img, (0, 0), (new_w, 80), color, -1)
    
    # Tulis Label Status
    font_scale = 1.0 if new_w > 400 else 0.6
    cv2.putText(display_img, label, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Tulis Nama File di Bawah
    filename = os.path.basename(file_path)
    cv2.putText(display_img, f"File: {filename}", (10, new_h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tampilkan Window
    window_name = "Sistem Inspeksi Ban Digital"
    cv2.imshow(window_name, display_img)
    
    print(f"--> Analisis Selesai: {label}")
    print("Tekan SPASI untuk pilih foto lain, atau 'q' untuk keluar.")
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        return False # Stop loop
    return True # Lanjut loop

# --- LOOP UTAMA ---
print("=========================================")
print("   APLIKASI DETEKSI BAN (MODE UPLOAD)    ")
print("=========================================")

while True:
    try:
        lanjut = proses_gambar()
        if not lanjut:
            break
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        break

cv2.destroyAllWindows()