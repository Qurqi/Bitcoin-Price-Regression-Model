import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf # Recommended to import tensorflow directly
import keras
from keras.layers import GRU, Dropout, SimpleRNN , Dense
from keras.models import Sequential
import keras_tuner

data = pd.read_csv('/Binance_1m_gmean.csv')
print(data.head())
#Remove Columns: Quote asset volume, Taker buy base asset volume, Taker buy quote asset volume, Ignore
kdata = data.drop(labels=["Unnamed: 0"],axis = 1)
#Partitiion size modifier
#Day = 1440, hour = 60, week = 10080
def choppy(data,mod):
  wdata_df = data
  return wdata_df
mod = "hour"
wdata_df = choppy(data,mod)
# Normalize data
norm = MinMaxScaler(feature_range=(1, 3))
ndata = norm.fit_transform(wdata_df)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
#Use 7 hours to predict the 8th
window_size = 7
X, y = create_sequences(ndata, window_size)
#break off 15% of data for validation
Vxsplit = int(0.15 * len(X))
Vysplit = int(0.15 * len(y))
X_val, X_rem = X[:Vxsplit], X[Vxsplit:]
y_val, y_rem = y[:Vysplit], y[Vysplit:]
splitx = int(0.8 * len(X_rem))
splity = int(0.8 * len(y_rem))
X_train, X_test = X_rem[:splitx], X_rem[splitx:]
y_train, y_test = y_rem[:splity], y_rem[splity:]
#Hyper parameter tuning
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 1, 5)):
      sw = i < hp.Int("num_layers", 1, 5) - 1
      model.add(GRU(units=hp.Int(f"units_{i}", min_value=50, max_value=200, step=1),activation="relu", return_sequences = sw))
      model.add(Dropout(hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.6, step=0.05)))
    model.add(Dense(6))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_logarithmic_error", metrics=["MeanSquaredLogarithmicError"],
    )
    return model
build_model(keras_tuner.HyperParameters())
directory = '/home/jobbinbobbin/Code/Python/Learn_Venv/'
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_MeanSquaredLogarithmicError",
    max_trials=10,
    executions_per_trial=4,
    overwrite=True,
    directory=directory,
    project_name="BTC_Models",
)
tuner.search(X_train, y_train, epochs=8, validation_data=(X_val, y_val))
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
best_model.save('BTC_Model_Test1.keras')
#########
#########
#########
#########
#########
##################
#########
##################
#########
##################
#########                                 BUILDING MODEL
##################
#########
##################
#########
##################
#########
#########
#########
#########
#########
predicted_gru = best_model.predict(X_test)
# Inverse transform the predicted and actual values
predicted_gru = norm.inverse_transform(predicted_gru)
y_test_actual = norm.inverse_transform(y_test)
# Plotting
pg = pd.DataFrame(predicted_gru)

ypa = pd.DataFrame(y_test_actual)

pg['Open'].plot(figsize=(5, 3),label='Predicted')
ypa['Open'].plot(figsize=(5, 3),label='Actual')
plt.title('Actual Vs. Predicted')
plt.xlabel('Time')
plt.ylabel('Open Value')
plt.legend()
plt.show()
r2d = 1 - np.sum((y_test_actual - predicted_gru)**2)/np.sum((y_test_actual - np.mean(y_test_actual))**2)
print("R^2 Test: ",r2d)
# try on other data set
