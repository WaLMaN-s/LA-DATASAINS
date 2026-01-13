import pandas as pd

# 1. Membaca file CSV
df = pd.read_csv("nilaiUTS_mahasiswa.csv")
print("=== Data Asli ===")
print(df)

# 2. Menghapus baris yang memiliki nilai kosong (NaN)
df_clean = df.dropna()
print("\n=== Data Setelah Dihapus Nilai Kosong ===")
print(df_clean)

# 3. Menampilkan nama kolom (cek ulang)
print("\n=== Nama Kolom pada Dataset ===")
print(df.columns)

# 4. Menghitung rata-rata, nilai tertinggi, dan nilai terendah berdasarkan mata kuliah
hasil_statistik = df_clean.groupby('mata_kuliah')['nilai_uts'].agg(['mean', 'max', 'min'])

print("\n=== Statistik Nilai UTS Berdasarkan Mata Kuliah ===")
print(hasil_statistik)

