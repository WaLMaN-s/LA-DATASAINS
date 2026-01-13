import pandas as pd

# Membuat DataFrame
data = {
    'nama': ['lutfi', 'basit', 'Citra'],
    'NPM': ['10123619', '10231002', '10231003'],
    'IPK': [3.75, 3.40, 3.90]
}

df = pd.DataFrame(data)

# Menampilkan DataFrame
print(df)
