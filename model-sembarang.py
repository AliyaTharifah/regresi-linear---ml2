import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data Jam Belajar (jam per minggu)
jam_belajar = np.array([2, 3, 5, 7, 8, 10, 12, 14, 15, 18])

# Data Skor Ujian (skala 0-100)
skor_ujian = np.array([50, 55, 60, 65, 68, 75, 80, 85, 90, 95])

print("Data Jam Belajar:", jam_belajar)
print("Data Skor Ujian:", skor_ujian)

plt.scatter(jam_belajar, skor_ujian, color='blue')
plt.title("Persebaran Data Jam Belajar vs Skor Ujian")
plt.xlabel("Jam Belajar")
plt.ylabel("Skor Ujian")
plt.show()

# Reshape data X menjadi array 2D
jam_belajar = jam_belajar.reshape(-1, 1)

# Inisialisasi dan melatih model
model = LinearRegression()
model.fit(jam_belajar, skor_ujian)

# Plot data asli
plt.scatter(jam_belajar, skor_ujian, color='blue', label='Data Asli')

# Plot garis regresi (prediksi)
plt.plot(jam_belajar, model.predict(jam_belajar), color='green', linewidth=2, label='Garis Regresi')

plt.title("Analisis Regresi: Jam Belajar vs Skor Ujian")
plt.xlabel("Jam Belajar")
plt.ylabel("Skor Ujian")
plt.legend()
plt.show()