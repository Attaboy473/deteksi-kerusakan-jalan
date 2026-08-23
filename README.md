# Deteksi Kerusakan Jalan Menggunakan YOLOv11

Aplikasi web untuk mendeteksi kerusakan permukaan jalan dengan model
YOLOv11n (nano) hasil training 50 epoch pada dataset 4 kelas:

- Lubang
- Retak Buaya
- Memanjang
- Melintang

## Fitur

- Upload gambar dengan visualisasi bounding box, tabel detail deteksi,
  dan tombol unduh gambar hasil.
- Pengambilan foto langsung dari kamera perangkat.
- Deteksi real-time melalui kamera dengan protokol WebRTC
  (inferensi non-blocking, resolusi 512 px).
- Pembacaan metadata EXIF dan lokasi GPS foto, lengkap dengan peta
  dan tautan Google Maps.

## Spesifikasi Teknis

| Parameter | Nilai |
| --- | --- |
| Arsitektur | YOLOv11n (nano) |
| Jumlah parameter | ±2.58 juta |
| Komputasi | ±6.3 GFLOPs |
| Ukuran input | 512 x 512 piksel |
| Confidence threshold | 0.423 (titik F1 terbaik hasil validasi) |

Confidence threshold dikunci pada nilai optimal hasil evaluasi, sehingga
pengguna tidak perlu mengatur parameter inferensi secara manual.

## Menjalankan Secara Lokal

1. Pastikan Python 3.11 terpasang.
2. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

   Untuk lingkungan lokal dengan GPU NVIDIA, paket `torch` dapat diganti
   dengan versi CUDA sesuai kebutuhan.

3. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

4. Buka `http://localhost:8501` di browser.

## Struktur Repositori

```
app.py                 Aplikasi Streamlit utama
best_yolo11n_jalanrusak_4kelas.pt   Bobot model YOLOv11n hasil training
requirements.txt       Dependensi Python
runtime.txt            Versi Python untuk Streamlit Community Cloud
.streamlit/config.toml Konfigurasi tema aplikasi
```

## Deploy ke Streamlit Community Cloud

Repositori ini siap dideploy langsung dari
[Streamlit Community Cloud](https://streamlit.io/cloud):

1. Masuk ke Streamlit Community Cloud dengan akun GitHub.
2. Pilih **New app**, lalu arahkan ke repositori ini.
3. File utama: `app.py`.
4. Deploy. HTTPS disediakan otomatis oleh platform.

Catatan: fitur deteksi real-time WebRTC memerlukan koneksi langsung
antar-perangkat. Jika jaringan tertentu memblokir paket WebRTC,
penambahan TURN server pada konfigurasi ICE di `app.py` dapat
dipertimbangkan.

## Lisensi

Proyek akademis (skripsi). Model dilatih pada dataset kerusakan jalan
4 kelas dengan pembagian train/valid/test.
