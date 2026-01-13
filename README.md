# LA-DATASAINS
# Ringkasan Singkat Konsep Data Mining

## 1. Regresi Linear dan Persamaan Dasarnya

Regresi linear adalah metode untuk memprediksi nilai numerik berdasarkan hubungan linear antar variabel.

**Persamaan dasar:**
- Regresi Linear Sederhana: **y = mx + b**
  - y = variabel target (yang diprediksi)
  - x = variabel input (prediktor)
  - m = slope (kemiringan garis)
  - b = intercept (titik potong sumbu y)

- Regresi Linear Berganda: **y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ**
  - Digunakan ketika ada banyak variabel input
  - Setiap x memiliki koefisien b yang menunjukkan pengaruhnya terhadap y

**Contoh:** Memprediksi harga rumah berdasarkan luas tanah, jumlah kamar, dan lokasi.

## 2. Klasifikasi dan Clustering

### Klasifikasi (Supervised Learning)
Memprediksi kategori/kelas dari data berdasarkan label yang sudah diketahui.

**Contoh algoritma:**
- **Decision Tree (C4.5, CART)** - Membuat pohon keputusan
- **Naive Bayes** - Menggunakan probabilitas
- **K-Nearest Neighbors (KNN)** - Berdasarkan kedekatan data
- **Support Vector Machine (SVM)** - Mencari pemisah optimal antar kelas
- **Random Forest** - Kombinasi banyak decision tree

**Contoh kasus:** Klasifikasi email spam/bukan spam, diagnosa penyakit, deteksi fraud.

### Clustering (Unsupervised Learning)
Mengelompokkan data berdasarkan kesamaan tanpa label yang sudah ada.

**Contoh algoritma:**
- **K-Means** - Membagi data menjadi k cluster berdasarkan centroid
- **Hierarchical Clustering** - Membuat hierarki cluster dalam bentuk dendrogram
- **DBSCAN** - Clustering berdasarkan kepadatan area
- **Mean Shift** - Menemukan centroid berdasarkan density

**Contoh kasus:** Segmentasi pelanggan, pengelompokan dokumen, analisis pola pembelian.

## 3. Clustering dengan K-Means

K-Means adalah algoritma clustering yang paling populer dan sederhana.

**Cara kerja:**
1. Tentukan jumlah cluster (k)
2. Inisialisasi k centroid secara random
3. Assign setiap data ke cluster dengan centroid terdekat
4. Hitung ulang centroid sebagai rata-rata posisi data dalam cluster
5. Ulangi langkah 3-4 hingga centroid tidak berubah

**Formula jarak (Euclidean):**
- d = √[(x₁-x₂)² + (y₁-y₂)²]

**Kelebihan:**
- Sederhana dan cepat
- Efisien untuk dataset besar

**Kekurangan:**
- Harus tentukan k di awal
- Sensitif terhadap inisialisasi awal
- Rentan terhadap outlier

**Contoh:** Mengelompokkan pelanggan menjadi 3 segmen (high, medium, low spender).

## 4. Klasifikasi dengan KNN

KNN (K-Nearest Neighbors) mengklasifikasikan data baru berdasarkan mayoritas kelas dari k tetangga terdekatnya.

**Cara kerja:**
1. Tentukan nilai k (jumlah tetangga)
2. Hitung jarak data baru ke semua data training
3. Pilih k data dengan jarak terdekat
4. Klasifikasi berdasarkan voting mayoritas dari k tetangga

**Pemilihan k:**
- k kecil (1-3): Sensitif terhadap noise
- k besar: Lebih smooth tapi bisa kehilangan detail
- Biasanya pilih k ganjil untuk hindari tie
- k optimal dicari dengan cross-validation

**Kelebihan:**
- Mudah dipahami dan diimplementasikan
- Tidak perlu training (lazy learning)
- Efektif untuk data multi-kelas

**Kekurangan:**
- Lambat untuk prediksi pada dataset besar
- Perlu normalisasi data
- Sensitif terhadap outlier

**Contoh:** Klasifikasi bunga iris berdasarkan panjang dan lebar kelopak dengan k=5.

## 5. Pipeline dan Hyperparameter Tuning

### Pipeline
Rangkaian tahapan pemrosesan data dan modeling yang terstruktur.

**Tahapan umum:**
1. **Data Cleaning** - Hapus missing values, duplikat, outlier
2. **Feature Engineering** - Buat fitur baru, transformasi data
3. **Feature Selection** - Pilih fitur yang relevan
4. **Normalization/Standardization** - Sesuaikan skala data
5. **Model Training** - Latih model machine learning
6. **Evaluation** - Evaluasi performa model

**Manfaat:**
- Workflow terorganisir dan reproducible
- Menghindari data leakage
- Mudah untuk maintenance dan deployment

### Hyperparameter Tuning
Proses mencari kombinasi parameter optimal untuk meningkatkan performa model.

**Perbedaan Parameter vs Hyperparameter:**
- **Parameter:** Dipelajari dari data (contoh: weight dalam neural network)
- **Hyperparameter:** Ditentukan sebelum training (contoh: k pada KNN, learning rate)

**Metode tuning:**

**Grid Search:**
- Mencoba semua kombinasi parameter dalam range tertentu
- Exhaustive tapi lambat
- Contoh: Coba k = [3, 5, 7, 9] dan distance = [euclidean, manhattan]

**Random Search:**
- Mencoba kombinasi parameter secara random
- Lebih cepat dari grid search
- Efektif untuk parameter yang banyak

**Cross-Validation:**
- Membagi data menjadi k folds
- Evaluasi setiap kombinasi parameter dengan CV
- Pilih parameter dengan performa terbaik

**Contoh hyperparameter:**
- KNN: nilai k, metrik jarak
- Decision Tree: kedalaman maksimal, minimum samples per leaf
- Neural Network: learning rate, jumlah layer, jumlah neurons

**Tujuan:**
- Meningkatkan akurasi model
- Menghindari overfitting
- Membuat model yang robust dan generalisasi baik

**Contoh praktis:**
Untuk KNN, coba k dari 1 sampai 20 dengan cross-validation, pilih k yang menghasilkan accuracy tertinggi.
