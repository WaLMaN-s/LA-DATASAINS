# Ringkasan Singkat Konsep Data Science

## 1. Regresi Linear dan Persamaan Dasarnya

Regresi linear adalah teknik supervised learning untuk memprediksi nilai numerik kontinu berdasarkan hubungan linear antar variabel.

**Persamaan dasar:**
- **Regresi Linear Sederhana:** y = mx + b
  - y = variabel target (yang diprediksi)
  - x = variabel independen (prediktor)
  - m = slope (koefisien regresi)
  - b = intercept (konstanta)

- **Regresi Linear Berganda:** y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
  - Digunakan ketika ada banyak variabel prediktor
  - Setiap x memiliki koefisien b yang menunjukkan kontribusinya

**Tujuan:** Menemukan garis yang paling fit dengan data dengan meminimalkan error (biasanya menggunakan metode Least Squares).

**Contoh aplikasi:** Memprediksi harga rumah berdasarkan luas bangunan, jumlah kamar, dan usia bangunan; memprediksi penjualan berdasarkan budget iklan; estimasi gaji berdasarkan pengalaman kerja.

**Evaluasi model:** Menggunakan metrik seperti R-squared (koefisien determinasi), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE).

## 2. Klasifikasi dan Clustering

Dua pendekatan utama dalam machine learning dengan tujuan berbeda.

### Klasifikasi (Supervised Learning)
Memprediksi kategori atau kelas dari data berdasarkan data training yang sudah memiliki label.

**Karakteristik:**
- Memerlukan labeled data untuk training
- Output berupa kelas diskrit (kategorikal)
- Tujuan: Belajar pola dari data untuk memprediksi kelas data baru

**Contoh algoritma klasifikasi:**
- **Logistic Regression** - Untuk klasifikasi binary
- **Decision Tree (C4.5, CART)** - Membuat pohon keputusan
- **Random Forest** - Ensemble dari banyak decision tree
- **Naive Bayes** - Berbasis probabilitas Bayes
- **K-Nearest Neighbors (KNN)** - Berdasarkan kedekatan data
- **Support Vector Machine (SVM)** - Mencari hyperplane pemisah optimal
- **Neural Networks** - Deep learning untuk pola kompleks

**Contoh kasus:** Deteksi spam email, diagnosa penyakit (positif/negatif), credit scoring (layak/tidak layak), image classification (kucing/anjing), sentiment analysis (positif/negatif/netral).

### Clustering (Unsupervised Learning)
Mengelompokkan data berdasarkan kesamaan karakteristik tanpa label yang telah ditentukan.

**Karakteristik:**
- Tidak memerlukan labeled data
- Output berupa kelompok/cluster
- Tujuan: Menemukan struktur tersembunyi dalam data

**Contoh algoritma clustering:**
- **K-Means** - Partitioning method berdasarkan centroid
- **Hierarchical Clustering** - Membuat hierarki cluster (dendrogram)
- **DBSCAN** - Density-based clustering
- **Gaussian Mixture Models (GMM)** - Probabilistic clustering
- **Mean Shift** - Mode-seeking algorithm

**Contoh kasus:** Customer segmentation, anomaly detection, document clustering, image segmentation, gene expression analysis, market basket analysis.

**Perbedaan utama:**
- Klasifikasi: Supervised (ada label) → Prediksi kelas
- Clustering: Unsupervised (tanpa label) → Eksplorasi pola

## 3. Clustering dengan K-Means

K-Means adalah algoritma clustering paling populer yang membagi data menjadi k kelompok berdasarkan kedekatan ke centroid.

**Cara kerja algoritma:**
1. **Inisialisasi:** Tentukan jumlah cluster k dan pilih k centroid awal secara random
2. **Assignment:** Assign setiap data point ke cluster dengan centroid terdekat
3. **Update:** Hitung ulang centroid sebagai mean dari semua data dalam cluster
4. **Iterasi:** Ulangi langkah 2-3 hingga centroid tidak berubah atau mencapai iterasi maksimum

**Formula jarak Euclidean:**
- d(p,q) = √[(x₁-x₂)² + (y₁-y₂)² + ... + (xₙ-yₙ)²]

**Menentukan k optimal:**
- **Elbow Method:** Plot WCSS vs k, cari titik "siku"
- **Silhouette Score:** Mengukur kualitas clustering (nilai -1 hingga 1)
- **Gap Statistic:** Membandingkan dengan distribusi referensi

**Kelebihan:**
- Sederhana dan mudah diimplementasikan
- Efisien untuk dataset besar (O(n×k×i))
- Scalable dan cepat konvergen

**Kekurangan:**
- Harus menentukan k di awal
- Sensitif terhadap inisialisasi centroid (gunakan K-Means++ untuk inisialisasi lebih baik)
- Rentan terhadap outlier
- Hanya cocok untuk cluster berbentuk spherical
- Mengasumsikan cluster memiliki ukuran dan variance yang sama

**Contoh aplikasi:** Segmentasi pelanggan berdasarkan perilaku belanja (high/medium/low value), pengelompokan dokumen berdasarkan topik, kompresi gambar dengan mengurangi jumlah warna.

## 4. Klasifikasi dengan KNN

K-Nearest Neighbors adalah algoritma klasifikasi non-parametrik yang mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekatnya.

**Cara kerja algoritma:**
1. **Input:** Data baru yang akan diklasifikasikan dan nilai k
2. **Kalkulasi jarak:** Hitung jarak dari data baru ke semua data training
3. **Seleksi neighbors:** Pilih k data dengan jarak terkecil
4. **Voting:** Untuk klasifikasi, gunakan majority voting; untuk regresi, gunakan rata-rata

**Metrik jarak yang umum:**
- **Euclidean:** d = √Σ(xᵢ-yᵢ)² (paling umum)
- **Manhattan:** d = Σ|xᵢ-yᵢ| (untuk data high-dimensional)
- **Minkowski:** generalisasi dari Euclidean dan Manhattan
- **Cosine Similarity:** untuk text classification

**Pemilihan nilai k:**
- **k terlalu kecil (k=1):** Sensitif terhadap noise, overfitting
- **k terlalu besar:** Under-sensitivity, underfitting
- **Rule of thumb:** k = √n (n = jumlah data training)
- **Best practice:** Gunakan k ganjil untuk binary classification (hindari tie)
- **Optimal k:** Cari dengan cross-validation

**Kelebihan:**
- Tidak ada fase training (lazy learning)
- Mudah dipahami dan diimplementasikan
- Efektif untuk data non-linear
- Dapat digunakan untuk klasifikasi dan regresi
- Tidak ada asumsi tentang distribusi data

**Kekurangan:**
- Komputasi lambat saat prediksi (O(n×d))
- Memerlukan banyak memori untuk menyimpan semua data training
- Sensitif terhadap skala fitur (perlu normalisasi/standardisasi)
- Performa menurun pada high-dimensional data (curse of dimensionality)
- Tidak cocok untuk imbalanced dataset

**Preprocessing penting:**
- **Normalisasi/Standardisasi:** Agar semua fitur memiliki skala yang sama
- **Feature selection:** Buang fitur yang tidak relevan

**Contoh aplikasi:** Sistem rekomendasi produk, diagnosa medis, pattern recognition, credit approval.

**Contoh kode Python:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load data
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model KNN dengan k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

## 5. Pipeline dan Hyperparameter Tuning

### Pipeline
Pipeline adalah rangkaian tahapan pemrosesan data dan modeling yang terstruktur dan otomatis dalam workflow data science.

**Tahapan umum dalam Pipeline:**
1. **Data Collection** - Mengumpulkan data dari berbagai sumber
2. **Data Cleaning** - Menangani missing values, duplikat, outlier
3. **Feature Engineering** - Membuat fitur baru, transformasi data
4. **Feature Selection** - Memilih fitur yang paling relevan
5. **Data Transformation** - Normalisasi, standardisasi, encoding
6. **Model Training** - Melatih model machine learning
7. **Model Evaluation** - Evaluasi performa model
8. **Model Deployment** - Deploy model ke production

**Manfaat menggunakan Pipeline:**
- **Reproducibility:** Workflow yang konsisten dan dapat diulang
- **Efficiency:** Otomasi proses preprocessing dan modeling
- **Menghindari data leakage:** Transformasi diterapkan dengan benar
- **Code organization:** Kode lebih bersih dan terstruktur
- **Easy deployment:** Mudah di-deploy sebagai satu unit

**Contoh kode Python:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Membuat pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Step 1: Normalisasi
    ('pca', PCA(n_components=2)),          # Step 2: Dimensionality reduction
    ('classifier', RandomForestClassifier()) # Step 3: Model
])

# Train pipeline (semua step dijalankan otomatis)
pipeline.fit(X_train, y_train)

# Prediksi
y_pred = pipeline.predict(X_test)

# Evaluasi
score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")
```

### Hyperparameter Tuning
Proses mencari kombinasi hyperparameter optimal untuk meningkatkan performa model machine learning.

**Perbedaan Parameter vs Hyperparameter:**
- **Parameter:** Nilai yang dipelajari oleh model dari data (contoh: weights dalam neural network, koefisien regresi)
- **Hyperparameter:** Konfigurasi yang diset sebelum training (contoh: k pada KNN, learning rate, jumlah trees dalam Random Forest)

**Metode Hyperparameter Tuning:**

**1. Grid Search**
- Mencoba semua kombinasi hyperparameter dalam grid yang ditentukan
- Exhaustive search (lengkap tapi lambat)
- Cocok untuk parameter space yang kecil

**2. Random Search**
- Mencoba kombinasi hyperparameter secara random
- Lebih efisien untuk parameter space yang besar
- Sering menemukan solusi baik lebih cepat dari Grid Search

**3. Bayesian Optimization**
- Menggunakan probabilitas untuk memilih hyperparameter berikutnya
- Lebih efisien dari Grid dan Random Search
- Cocok untuk expensive model training

**Teknik validasi:**
- **K-Fold Cross-Validation:** Membagi data menjadi k folds untuk evaluasi robust
- **Stratified K-Fold:** Mempertahankan proporsi kelas di setiap fold
- **Time Series Split:** Untuk data time series

**Contoh hyperparameter:**
- **KNN:** n_neighbors, metric (euclidean/manhattan), weights
- **Decision Tree:** max_depth, min_samples_split, criterion (gini/entropy)
- **Random Forest:** n_estimators, max_depth, min_samples_leaf
- **Neural Network:** learning_rate, batch_size, epochs, optimizer
- **SVM:** C, kernel, gamma

**Tujuan Hyperparameter Tuning:**
- Meningkatkan akurasi dan performa model
- Menghindari overfitting dan underfitting
- Membuat model yang dapat generalisasi dengan baik
- Menemukan sweet spot antara bias dan variance

**Contoh kode Python - Grid Search:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,              # Use all CPU cores
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")
```

**Contoh kode Python - Random Search:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20)
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,              # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

**Contoh kode Python - Pipeline + Grid Search:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Parameter grid for pipeline (gunakan __ untuk akses step)
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto', 0.1, 1]
}

# Grid search on pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.3f}")
```

---

## Ringkasan Library Python untuk Data Science

**Data manipulation:**
- `pandas` - DataFrame manipulation
- `numpy` - Numerical computing

**Visualization:**
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots

**Machine Learning:**
- `scikit-learn` - Algoritma ML lengkap
- `tensorflow/keras` - Deep learning
- `pytorch` - Deep learning framework
- `xgboost` - Gradient boosting

**Model evaluation:**
- `sklearn.metrics` - Evaluation metrics
- `sklearn.model_selection` - Cross-validation, train-test split
