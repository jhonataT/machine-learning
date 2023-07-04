# pip install matplotlib
# pip install pandas
# pip install -U scikit-learn

from matplotlib import pyplot as plt
import pandas as pd
import pylab as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

df = pd.read_csv('FuelConsumptionCo2.csv');

engines = df[['ENGINESIZE']];
co2 = df[['CO2EMISSIONS']];

trainEngines, testEngines, trainCo2, testCo2 = train_test_split(
    engines,
    co2,
    test_size=0.2,
    random_state=42
);

# Plot Scatter Chart:

# plt.scatter(trainEngines, trainCo2, color='red');
# plt.xlabel('Motor');
# plt.ylabel('Emissão de CO2');
# plt.show();

# Create new linear regression model:
model = linear_model.LinearRegression();

# Traning the model:
model.fit(trainEngines, trainCo2);

# print('(A) Intercept: ', model.intercept_);
# print('(B) Coef (Inclinação): ', model.coef_);

# Show train Straight:

# plt.scatter(trainEngines, trainCo2, color='gray ');
# plt.plot(trainEngines, model.coef_[0][0]*trainEngines + model.intercept_[0], '-r');
# plt.ylabel('Emissão de CO2');
# plt.xlabel('Motores');
# plt.show();

# Predict using model:
co2Predict = model.predict(testEngines);

# Show test Straight:

# plt.scatter(testEngines, testCo2, color='gray');
# plt.plot(testEngines, model.coef_[0][0] * testEngines + model.intercept_[0], '-r');
# plt.ylabel('Emissão de CO2');
# plt.xlabel('Motores');
# plt.show();

# Metrics to evaluate the model:

print('Soma dos Erros ao Quadrado (SSE): %.2f ' % np.sum((co2Predict - testCo2)**2));
print('Erro Quadrático Médio (MSE): %.2f ' % mean_squared_error(testCo2, co2Predict));
print('Erro Médio Absoluto (MAE): %.2f ' % mean_absolute_error(testCo2, co2Predict));
print('Raiz do Erro Quadrático Médio (RMSE): %.2f ' % sqrt(mean_squared_error(testCo2, co2Predict)));
print('R2-core: %.2f ' % r2_score(co2Predict, testCo2));