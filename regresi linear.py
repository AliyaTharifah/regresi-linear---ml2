import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error 

df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

print(cdf.describe())

plt.figure(figsize=(8,6))
sns.heatmap(cdf.corr(), annot=True, cmap='coolwarm')
plt.title("Matriks Korelasi Variabel")
plt.show()

# Membagi data secara acak
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Inisialisasi model
regr = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x, train_y)

print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# prediksi pada data test
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

# Menghitung metrik evaluasi
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='blue', label='Actual Data')
plt.plot(test_x, test_y_hat, '-r', label='Regression Line')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Evaluasi Regresi pada Data Test")
plt.legend()
plt.show()