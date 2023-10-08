import warnings

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, GRU
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from pandas import read_csv

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from math import sqrt

timeSeries = None


def withANN():
    data = pd.read_csv("dsc_fc_summed_spectra_2022_v01.csv", \
                       delimiter=',', parse_dates=[0], \
                       infer_datetime_format=True, na_values='0', \
                       header=None)


def predictionCSVLSTM(self):
    warnings.filterwarnings("ignore")
    print("#### -> predictionCSV <- ####")

    from keras.utils import np_utils

    dataset = pd.read_csv("SonucSaatlikTumRuzgar2.csv", delimiter=',')

    # input verillerinin data türlerinin kontrolü
    print(dataset.dtypes)
    print(dataset.head())

    dataset['dateColumn'] = pd.to_datetime(dataset.dateColumn, format='%Y-%m-%d %H:%M')
    data = dataset.drop(['dateColumn'], axis=1)
    data.index = dataset.dateColumn

    # input verillerinin data türlerinin kontrolü
    print(data.dtypes)
    print(data.head())

    # input verilerinin normallestirilmesi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    scaled = pd.DataFrame(scaled)

    print("scaled.head(4)")
    print(scaled.head(4))

    def create_ts_data(dataset, lookback=1, predicted_col=3):
        temp = dataset.copy()

        temp["id"] = range(1, len(temp) + 1)
        temp = temp.iloc[:-lookback, :]
        temp.set_index('id', inplace=True)
        predicted_value = dataset.copy()
        predicted_value = predicted_value.iloc[lookback:, predicted_col]
        predicted_value.columns = ["Predcited"]
        predicted_value = pd.DataFrame(predicted_value)

        predicted_value["id"] = range(1, len(predicted_value) + 1)
        predicted_value.set_index('id', inplace=True)
        final_df = pd.concat([temp, predicted_value], axis=1)
        # final_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)', 'var8(t-1)','var1(t)']
        # final_df.set_index('Date', inplace=True)
        return final_df

    reframed_df = create_ts_data(scaled, 1, 2)
    reframed_df.fillna(0, inplace=True)

    reframed_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', ]
    print("reframed_df.head(4)")
    print(reframed_df.head(4))

    # dataların test ve eğitim olarak ayrılması
    values = reframed_df.values
    training_sample = int(len(dataset) * 0.7)
    train = values[:training_sample, :]
    test = values[training_sample:, :]
    # dataların test ve eğitim olarak ayrılması
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print("Dizilerin Boyutları")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model_lstm = Sequential()
    model_lstm.add(LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=50, return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=1))

    model_lstm.compile(loss='mae', optimizer='adam')
    model_lstm.summary()

    lstm_history = model_lstm.fit(train_X, train_y, epochs=100, validation_data=(test_X, test_y), batch_size=64,
                                  shuffle=False)

    pred_y = model_lstm.predict(test_X)
    plt.plot(lstm_history.history['loss'], label='lstm train', color='brown')
    plt.plot(lstm_history.history['val_loss'], label='lstm test', color='blue')
    plt.title("YSA Model Kayıpları Rüzgar (Basınç-Nem)")
    plt.legend()
    plt.show()

    plt.rcParams['figure.figsize'] = (15, 5)

    MSE = mean_squared_error(test_y, pred_y)
    R2 = r2_score(test_y, pred_y)
    RMSE = sqrt(mean_squared_error(test_y, pred_y))
    MAE = mean_absolute_error(test_y, pred_y)

    print("MSE  :", MSE)
    print("r2_score :", R2)
    print("RMSE :", RMSE)
    print("MAE  :", MAE)

    # Test verisi tahmini, gerçek veri grafiği
    plt.plot(test_y, label='Actual')
    plt.plot(pred_y, label='Predicted')
    plt.legend()
    plt.title("Rüzgar (Basınç-Nem) Tahmin / Gerçek Test Datası ")
    plt.show()

    # Tüm verinin görselleştirilmesi
    tra = np.concatenate([train_X, test_X])
    tes = np.concatenate([train_y, test_y])
    fp = model_lstm.predict(tra)
    plt.plot(tes, label='Actual')
    plt.plot(fp, label='Predicted')
    plt.title("Rüzgar (Basınç-Nem) Tahmin / Gerçek  Bütün Data ")
    plt.legend()
    plt.show()


def predictionLSTM2():
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    warnings.filterwarnings("ignore")
    print("#### -> predictionCSV <- ####")
    print("#### -> dataset with date <- ####")

    from keras.utils import np_utils

    # dataset  = pd.read_csv("data/dsc_fc_summed_spectra_2022_v01.csv", delimiter=',', infer_datetime_format=True, na_values='0', header=1)
    dataset = pd.read_csv("data/dsc_fc_summed_spectra_2022_v01.csv", delimiter=',')

    # input verillerinin data türlerinin kontrolü
    print(dataset.dtypes)
    print(dataset.head())

    print("#### -> dataset no date <- ####")

    dataset['dateColumn'] = pd.to_datetime(dataset.dateColumn, format='%Y-%m-%d %H:%M:%S')
    data = dataset.drop(['dateColumn'], axis=1)
    data.index = dataset.dateColumn

    # input verillerinin data türlerinin kontrolü
    print(data.dtypes)
    print(data.head())

    # input verilerinin normallestirilmesi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    scaled = pd.DataFrame(scaled)

    print("scaled.head(10)")
    print(scaled.head(10))

    y = scaled["vx"].to_numpy()

    columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    x = scaled[columns]

    plt.plot(y)
    plt.show()

    print(x.shape)

    x = x.reshape(x.shape[0], x.shape[1], 1)
    print("x:", x.shape, "y:", y.shape)

    in_dim = (x.shape[1], x.shape[2])
    out_dim = y.shape[1]
    print(in_dim)
    print(out_dim)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
    print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)

    model = Sequential()
    model.add(LSTM(64, input_shape=in_dim, activation="relu"))
    model.add(Dense(out_dim))
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    model.fit(xtrain, ytrain, epochs=100, batch_size=12, verbose=0)

    ypred = model.predict(xtest)
    print("y1 MSE:%.4f" % mean_squared_error(ytest[:, 0], ypred[:, 0]))
    print("y2 MSE:%.4f" % mean_squared_error(ytest[:, 1], ypred[:, 1]))

    x_ax = range(len(xtest))
    plt.title("LSTM multi-output prediction")
    plt.scatter(x_ax, ytest[:, 0], s=6, label="y1-test")
    plt.plot(x_ax, ypred[:, 0], label="y1-pred")
    plt.scatter(x_ax, ytest[:, 1], s=6, label="y2-test")
    plt.plot(x_ax, ypred[:, 1], label="y2-pred")
    plt.legend()
    plt.show()

def predictionCSVGRU():

    warnings.filterwarnings("ignore")
    print("#### -> predictionCSV <- ####")
    print("#### -> dataset with date <- ####")

    from keras.utils import np_utils

    # dataset  = pd.read_csv("data/dsc_fc_summed_spectra_2022_v01.csv", delimiter=',', infer_datetime_format=True, na_values='0', header=1)
    dataset = pd.read_csv("data/dsc_fc_summed_spectra_2022_v01.csv", delimiter=',')

    # input verillerinin data türlerinin kontrolü
    #print(dataset.dtypes)
    print(dataset.head(5))

    print("#### -> dataset no date <- ####")

    kp = pd.read_csv("data/kp_2016_2023.csv", delimiter=',')
    print("kp.head(5)")
    print(kp.head(5))

    print("#### -> dataset merged <- ####")
    datamerged = pd.merge(dataset, kp, on="dateColumn")
    print(datamerged.head(5))

    datamerged['dateColumn'] = pd.to_datetime(datamerged.dateColumn, format='%Y-%m-%d %H:%M:%S')
    data = datamerged.drop(['dateColumn'], axis=1)
    data.index = datamerged.dateColumn

    print(datamerged.head(5))

    # input verilerinin normallestirilmesi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    scaled = pd.DataFrame(scaled)

    print("scaled.head()")
    print(scaled.head(5))

    def create_ts_data(dataset, lookback=1, predicted_col=53):
        temp = dataset.copy()

        temp["id"] = range(1, len(temp) + 1)
        temp = temp.iloc[:-lookback, :]
        temp.set_index('id', inplace=True)
        predicted_value = dataset.copy()
        predicted_value = predicted_value.iloc[lookback:, predicted_col]
        predicted_value.columns = ["Predcited"]
        predicted_value = pd.DataFrame(predicted_value)

        predicted_value["id"] = range(1, len(predicted_value) + 1)
        predicted_value.set_index('id', inplace=True)
        #final_df = pd.concat([temp, predicted_value], axis=1)
        final_df = pd.concat([temp, predicted_value], axis=1)
        # final_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)', 'var8(t-1)','var1(t)']
        # final_df.set_index('Date', inplace=True)
        return final_df

    reframed_df = create_ts_data(scaled, 1, 53 )
    #reframed_df = scaled.copy();
    reframed_df.fillna(0, inplace=True)

    print("#### -> dataset reframed_df <- ####")

    #print("reframed_df.head(5)")
    #print(reframed_df.head(5))

    #reframed_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', ]

    print("reframed_df.head(5)")
    print(reframed_df.head(5))

    print("reframed_df.tail(5)")
    print(reframed_df.tail(5))

    # input dataların test ve eğitim olarak ayrılması
    values = reframed_df.values
    training_sample = int(len(dataset) * 0.75)
    train = values[:training_sample, :]
    test = values[training_sample:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print("Shapes")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model_gru = Sequential()
    model_gru.add(GRU(75, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model_gru.add(GRU(units=30, return_sequences=True))
    model_gru.add(GRU(units=30))
    model_gru.add(Dense(units=1))

    model_gru.compile(loss='mae', optimizer='adam')
    model_gru.summary()

    # fit network
    gru_history = model_gru.fit(train_X, train_y, epochs=100, validation_data=(test_X, test_y), batch_size=64,
                                shuffle=False)

    pred_y = model_gru.predict(test_X)

    # dont run this cell if you are running this cell than add "validation_data=(test_X, test_y)" in model_gru.fit()
    plt.plot(gru_history.history['loss'], label='GRU train', color='brown')
    plt.plot(gru_history.history['val_loss'], label='GRU test', color='blue')
    plt.title("YSA Model Kayıpları GRU Method Sun Rüzgar ")
    plt.legend()
    plt.show()

    plt.rcParams['figure.figsize'] = (15, 5)

    ##from sklearn.metrics import *
    from math import sqrt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score

    MSE = mean_squared_error(test_y, pred_y)
    R2 = r2_score(test_y, pred_y)
    RMSE = sqrt(mean_squared_error(test_y, pred_y))
    MAE = mean_absolute_error(test_y, pred_y)

    print("MSE  :", MSE)
    print("r2_score :", R2)
    print("RMSE :", RMSE)
    print("MAE  :", MAE)

    # Test verisi tahmini, gerçek veri grafiği
    plt.plot(test_y, label='Actual')
    plt.plot(pred_y, label='Predicted')
    plt.legend()
    plt.title("GRU Metod  Tahmin / Gerçek Test Datası ")
    plt.show()

    # Tüm verinin görselleştirilmesi
    tra = np.concatenate([train_X, test_X])
    tes = np.concatenate([train_y, test_y])
    fp = model_gru.predict(tra)
    plt.plot(tes, label='Actual')
    plt.plot(fp, label='Predicted')
    plt.title("GRU Metod  Tahmin / Gerçek  Bütün Data ")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # withTimeSeries()
    predictionCSVGRU()
    #predictionLSTM2()
