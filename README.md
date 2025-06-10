# ✋ Gesture Link

***Menghubungkan Gerakan dengan Bahasa***

**Gesture Link** adalah aplikasi web interaktif yang mengenali gesture Bahasa Isyarat Indonesia (BISINDO) menggunakan **MediaPipe** dan model klasifikasi berbasis landmark tangan. Aplikasi ini memiliki tiga fitur utama: **Deteksi Kalimat (Coba Sekarang)**, **Belajar Alfabet & Angka BISINDO**, dan **Quiz** untuk menguji kemampuan.

---

## 📸 Screenshot Aplikasi

| Halaman Beranda                  | Halaman Belajar                   | Halaman Quiz                     |
| -------------------------------- | --------------------------------- | -------------------------------- |
| ![](static/screenshots/home.png) | ![](static/screenshots/learn.png) | ![](static/screenshots/quiz.png) |

---

## 🚀 Fitur Aplikasi

### 1. **Coba Sekarang**

* Mendeteksi gesture tangan secara real-time dan mengonversi menjadi huruf dan angka.
* Dapat membentuk **kalimat** dari rangkaian isyarat yang dikenali.

### 2. **Modul Belajar**

* Menyediakan 37 modul (A–Z dan 0–10).
* Setiap modul menyediakan contoh gesture dan sistem validasi otomatis apakah gesture pengguna sudah benar.
* Progres disimpan di sesi dan ditandai dengan ✔️.

### 3. **Quiz**

* Berisi 10 soal acak dari huruf dan angka.
* Waktu menjawab maksimal 10 detik per soal.
* Skor ditampilkan di akhir quiz.

---

## 🧰 Library yang Digunakan

Berikut adalah daftar pustaka utama dari proses awal hingga deployment web:

| Tahapan                        | Library                                         |
| ------------------------------ | ----------------------------------------------- |
| **Perekaman & Ekstraksi Data** | `OpenCV`, `MediaPipe`, `csv`, `os`              |
| **Pelatihan Model**            | `TensorFlow`, `NumPy`, `Pandas`, `scikit-learn` |
| **Aplikasi Web**               | `Flask`, `Jinja2`, `Werkzeug`, `HTML/CSS/JS`    |
| **Suara Notifikasi**           | `pygame.mixer`                                  |

---

## 📁 Struktur Proyek

```
gesture-link/
├── dataset/
│   ├── dataset_landmark_csv/
│   └── labels.txt
├── model/
|   ├── training
|   |   └── Training_Bisindo_landmark.ipynb
│   ├── best_model.h5
│   └── label_classes.npy
├── static/
|   ├── img/
|   ├── img_signs/
│   ├── style.css
│   └── landing.css
├── templates/
│   ├── index.html
│   ├── landing.html
│   ├── learn.html
│   ├── learn_single.html
│   └── quiz.html
├── utils/
│   ├── beep.mp3
│   └── sound.py
├── app.py                # Aplikasi web Flask
├── collect_landmark_data.py  # Rekam data baru
├── realtime_prediction.py    # Deteksi real-time lokal
└── requirements.txt
```

---

## 💻 Cara Menjalankan Proyek

### 1. Clone dan setup environment

```bash
git clone https://github.com/GestureLink/gesture-link.git
cd gesture-link
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Menjalankan deteksi lokal (non-web)

```bash
python realtime_prediction.py
```

### 3. Menjalankan aplikasi web Flask

```bash
python app.py
```

Akses di browser melalui `http://127.0.0.1:5000`.

---

## 🤝 Cara Berkontribusi: Menambahkan Data Baru

1. Jalankan:

```bash
python collect_landmark_data.py
```

2. Masukkan:

   * Label gesture (contoh: `A`, `B`, `5`)
   * Jumlah data yang ingin direkam (contoh: `30`)
3. Sistem akan:

   * Menunggu tangan terdeteksi oleh MediaPipe
   * Countdown 10 detik
   * Menyimpan `.csv` dan gambar gesture ke folder `dataset/`

📌 *Data tidak akan menimpa data sebelumnya.*

---

## 👨‍💻 Kontributor

* @LAI25-SS040
* Auliya
* Cindy 
* Rozaq
* Verzha

---

Jika kamu memiliki saran, fitur baru, atau ingin membantu memperluas dataset BISINDO — **jangan ragu untuk fork dan pull request!** 🙌
