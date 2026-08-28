# Rancang Bangun Smart Vision Satria Navigait

## Berbasis Artificial Intelligence dengan Output Bone Conduction sebagai Solusi Navigasi Mandiri Tunanetra Secara Real-Time

## 📌 Deskripsi Proyek

**Smart Vision Satria Navigait** merupakan sebuah sistem berbasis
**Artificial Intelligence (AI)** yang dirancang sebagai solusi teknologi
untuk mendukung **navigasi mandiri tunanetra secara real-time**.

Sistem memanfaatkan **Raspberry Pi** sebagai komponen pemrosesan utama
untuk menerima dan mengolah informasi visual dari kamera. Data visual
tersebut diproses menggunakan model AI untuk mengenali kondisi
lingkungan yang relevan dengan kebutuhan navigasi.

Informasi hasil pemrosesan kemudian diterjemahkan menjadi instruksi atau
peringatan suara yang disampaikan kepada pengguna melalui **bone
conduction**. Teknologi ini memungkinkan informasi audio diterima
melalui getaran pada tulang di sekitar telinga, sehingga telinga tetap
relatif terbuka terhadap suara lingkungan sekitar.

------------------------------------------------------------------------

## 🎯 Tujuan

Proyek ini bertujuan untuk:

1.  Mengembangkan perangkat bantu navigasi berbasis AI bagi penyandang
    tunanetra.
2.  Memproses informasi lingkungan secara **real-time** menggunakan
    Raspberry Pi.
3.  Mengidentifikasi kondisi lingkungan yang penting untuk mendukung
    pengambilan keputusan saat berjalan.
4.  Memberikan informasi navigasi melalui output audio.
5.  Menggunakan teknologi **bone conduction** agar pengguna tetap dapat
    menerima informasi dari lingkungan sekitar.
6.  Mendukung kemandirian pengguna dalam melakukan navigasi di
    lingkungan sekitar.

------------------------------------------------------------------------

## 🧠 Konsep Sistem

Secara umum, sistem bekerja dengan alur:

``` text
Kamera
   │
   ▼
Pengambilan Citra
   │
   ▼
Raspberry Pi
   │
   ▼
Pemrosesan Artificial Intelligence
   │
   ▼
Analisis Kondisi Lingkungan
   │
   ▼
Decision Making
   │
   ▼
Instruksi / Peringatan Suara
   │
   ▼
Bone Conduction
   │
   ▼
Pengguna Tunanetra
```

------------------------------------------------------------------------

## ⚙️ Komponen Utama

### 1. Raspberry Pi

Raspberry Pi digunakan sebagai **komponen pemrosesan utama** pada
perangkat. Raspberry Pi bertugas menjalankan program, memproses data
kamera, menjalankan model AI, melakukan pengambilan keputusan, dan
mengatur keluaran audio.

### 2. Kamera

Kamera digunakan sebagai sumber data visual. Citra dari lingkungan
sekitar ditangkap secara langsung dan diteruskan ke Raspberry Pi untuk
diproses oleh sistem AI.

### 3. Artificial Intelligence

Model AI digunakan untuk menganalisis citra dan memperoleh informasi
mengenai kondisi lingkungan yang diperlukan dalam proses navigasi.

Pemrosesan AI dilakukan secara lokal pada Raspberry Pi sehingga sistem
dapat memberikan respons secara real-time tanpa bergantung sepenuhnya
pada pemrosesan cloud.

### 4. Bone Conduction

Bone conduction digunakan sebagai media keluaran suara. Instruksi atau
peringatan yang dihasilkan sistem dikirimkan kepada pengguna dalam
bentuk suara melalui mekanisme getaran.

Penggunaan bone conduction ditujukan agar pengguna tetap dapat mendengar
suara lingkungan sekitar ketika menerima informasi dari perangkat.

------------------------------------------------------------------------

## 🔄 Alur Kerja Sistem

### Tahap 1 --- Akuisisi Data

Kamera menangkap kondisi lingkungan di depan pengguna secara real-time.

### Tahap 2 --- Pemrosesan Citra

Data visual diteruskan ke Raspberry Pi untuk dilakukan preprocessing dan
analisis menggunakan model AI.

### Tahap 3 --- Analisis AI

Model AI melakukan analisis terhadap citra untuk memperoleh informasi
mengenai kondisi lingkungan yang relevan dengan navigasi.

### Tahap 4 --- Decision Making

Hasil analisis AI digunakan oleh sistem untuk menentukan kondisi dan
respons yang sesuai.

Contoh keluaran keputusan sistem meliputi:

-   Jalan masih dapat dilalui secara lurus.
-   Terdapat orang pada area navigasi.
-   Pengguna perlu berhenti.
-   Pengguna perlu melakukan perubahan arah.
-   Pengguna perlu melakukan putar balik.

### Tahap 5 --- Audio Guidance

Keputusan sistem diterjemahkan menjadi instruksi suara.

### Tahap 6 --- Bone Conduction Output

Instruksi suara disampaikan kepada pengguna melalui perangkat bone
conduction dalam bentuk getaran sehingga pengguna dapat menerima
informasi navigasi tanpa harus menutup telinga dari suara lingkungan.

------------------------------------------------------------------------

## 🗺️ Sistem Navigasi

Sistem dirancang untuk membantu pengguna memahami kondisi area di depan
secara lebih terstruktur.

Analisis area dilakukan berdasarkan hasil segmentasi dan pembagian area
pengamatan. Informasi tersebut digunakan sebagai dasar pengambilan
keputusan oleh sistem.

Salah satu parameter yang digunakan adalah **B4**, yaitu bagian area
pengamatan yang digunakan sebagai indikator kondisi jalur di depan
pengguna.

Contoh konsep keputusan:

``` text
             HASIL SEGMENTASI
                    │
          ┌─────────┴─────────┐
          │                   │
         B4              PERSON C3?
          │                   │
          ▼                   ▼
       B4 > 8%?              TRUE?
          │                   │
     ┌────┴────┐         ┌────┴────┐
     │         │         │         │
   TRUE      FALSE     TRUE      FALSE
     │         │         │         │
     ▼         ▼         ▼         ▼
   LURUS    BERHENTI  ADA ORANG  EVALUASI
```

> **Catatan:** Aturan keputusan dapat dikembangkan sesuai hasil
> pengujian dan validasi sistem.

------------------------------------------------------------------------

## 🔊 Output Audio

Sistem menyediakan beberapa instruksi audio yang digunakan sebagai
respons terhadap kondisi navigasi.

File audio yang digunakan pada sistem:

``` text
audio/
├── ada_orang.mp3
├── belok kanan.mp3
├── belok kiri.mp3
├── berhenti.mp3
├── lurus.mp3
└── putar balik.mp3
```

Contoh pemetaan:

  Kondisi       Output
  ------------- -------------------
  Jalan lurus   `lurus.mp3`
  Ada orang     `ada_orang.mp3`
  Berhenti      `berhenti.mp3`
  Belok kanan   `belok kanan.mp3`
  Belok kiri    `belok kiri.mp3`
  Putar balik   `putar balik.mp3`

------------------------------------------------------------------------

## 🧩 Struktur Repository

Struktur repository secara umum:

``` text
sidewalk_edgedevice/
│
├── audio/
│   ├── ada_orang.mp3
│   ├── belok kanan.mp3
│   ├── belok kiri.mp3
│   ├── berhenti.mp3
│   ├── lurus.mp3
│   └── putar balik.mp3
│
├── model_tobias/
│   └── sidewalk_tobias.onnx
│
├── src/
│   └── sidewalk_main.py
│
├── test_data/
│
├── requirement.txt
│
└── README.md
```

------------------------------------------------------------------------

## 💻 Teknologi

Teknologi utama yang digunakan:

-   **Python**
-   **Raspberry Pi**
-   **OpenCV**
-   **NumPy**
-   **ONNX Runtime**
-   **SegFormer**
-   **ONNX**
-   **Pygame**
-   **Bone Conduction**

------------------------------------------------------------------------

## 📦 Instalasi

Clone repository:

``` bash
git clone <URL_REPOSITORY>
cd sidewalk_edgedevice
```

Buat virtual environment:

``` bash
python3 -m venv sidewalk_env
```

Aktifkan environment:

``` bash
source sidewalk_env/bin/activate
```

Install library:

``` bash
pip install -r requirement.txt
```

------------------------------------------------------------------------

## ▶️ Menjalankan Sistem

Setelah seluruh komponen terhubung dan library telah terpasang:

``` bash
python src/sidewalk_main.py
```

Sistem kemudian akan:

1.  Memuat konfigurasi audio.
2.  Memuat model AI.
3.  Mengaktifkan kamera.
4.  Mengambil frame secara real-time.
5.  Melakukan preprocessing.
6.  Menjalankan inferensi AI.
7.  Melakukan analisis hasil segmentasi.
8.  Menentukan keputusan navigasi.
9.  Memutar instruksi audio.
10. Mengirimkan output suara melalui bone conduction.

------------------------------------------------------------------------

## 🚀 Keunggulan Konsep

Beberapa karakteristik utama sistem:

-   **Real-time processing** menggunakan Raspberry Pi.
-   **Pemrosesan lokal** tanpa harus mengirim seluruh data visual ke
    server.
-   Memanfaatkan **Artificial Intelligence** untuk memahami lingkungan.
-   Memberikan **audio guidance** sebagai informasi navigasi.
-   Menggunakan **bone conduction** sebagai media output.
-   Dirancang untuk mendukung **navigasi mandiri tunanetra**.

------------------------------------------------------------------------

## 🧪 Pengujian

Pengujian sistem dilakukan untuk mengevaluasi kemampuan perangkat dalam:

-   Mengenali kondisi jalur.
-   Mengenali keberadaan orang.
-   Menentukan posisi objek pada area pengamatan.
-   Menghasilkan keputusan navigasi.
-   Memberikan output audio sesuai keputusan.
-   Memberikan respons secara real-time.

Parameter pengujian dapat mencakup:

  -----------------------------------------------------------------------
  Parameter                           Keterangan
  ----------------------------------- -----------------------------------
  Akurasi deteksi                     Kesesuaian hasil AI dengan kondisi
                                      sebenarnya

  Waktu respons                       Waktu dari input kamera hingga
                                      output audio

  Keberhasilan audio                  Kesesuaian instruksi dengan
                                      keputusan

  Konsistensi sistem                  Stabilitas keputusan pada beberapa
                                      frame

  Real-time performance               Kemampuan sistem memproses data
                                      secara langsung
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 👥 Pengembangan

Proyek ini dikembangkan sebagai bagian dari kegiatan **Olimpiade**
dengan fokus pada penerapan Artificial Intelligence, embedded system,
dan teknologi bantu untuk menyelesaikan permasalahan navigasi.

Nama sistem:

> **Smart Vision Satria Navigait**

Konsep utama:

> **AI Vision → Raspberry Pi → Decision Making → Bone Conduction**

------------------------------------------------------------------------

## 📜 Lisensi

Repository ini dibuat untuk kebutuhan pengembangan dan dokumentasi
proyek **Smart Vision Satria Navigait**.

Lisensi dapat ditentukan sesuai kebutuhan tim pengembang.
