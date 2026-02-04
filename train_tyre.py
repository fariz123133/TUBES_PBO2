import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- KONFIGURASI ---
# Pastikan nama folder ini sesuai dengan Langkah 1
base_dir = 'dataset' 

# Cek apakah folder terbaca
if not os.path.exists(base_dir):
    print(f"ERROR: Folder '{base_dir}' tidak ditemukan! Cek Langkah 1 lagi.")
    exit()

# Setup Image Generator (Agar data variatif & normalisasi)
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Ubah nilai pixel jadi 0-1
    rotation_range=20,    # Putar gambar sedikit (simulasi posisi kamera miring)
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 20% data diambil untuk tes otomatis
)

# Load Data Training
print("Memuat Data Training...")
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150), # Kita resize ke 150x150 biar ringan
    batch_size=32,
    class_mode='binary',    # Binary karena cuma 2: Good vs Defective
    subset='training'
)

# Load Data Validasi
print("Memuat Data Validasi...")
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Cek Label (PENTING UNTUK NANTI DETEKSI)
print("\n--- PENTING: CATAT INI ---")
print(f"Label Mapping: {train_generator.class_indices}")
print("--------------------------\n")

# Bangun Model AI (CNN)
model = Sequential([
    # Layer 1: Deteksi fitur dasar (garis/lengkungan)
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    # Layer 2: Deteksi fitur kompleks (pola ulir)
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Layer 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(), # Ratakan jadi 1 baris
    Dense(512, activation='relu'),
    Dropout(0.5), # Matikan 50% neuron acak biar tidak menghapal
    Dense(1, activation='sigmoid') # Output 0 atau 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Mulai Training
print("Mulai melatih model (Proses ini makan waktu tergantung spek laptop)...")
history = model.fit(
    train_generator,
    epochs=15, # Ulangi belajar 15 kali
    validation_data=validation_generator
)

# Simpan Model
model.save('model_ban.h5')
print("\nSUKSES! File 'model_ban.h5' berhasil dibuat.")