import pandas as pd
import streamlit as st
import pickle
import numpy as np
import sklearn
import torch
import torch.nn as nn
from classes.lstm import Optimization
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from classes.transformer import Time2Vector, MultiAttention, SingleAttention, TransformerEncoder

LAG_FEATURES = 10

dataTrain = pd.read_csv("./data/DailyDelhiClimateTrain.csv")
dataTest = pd.read_csv("./data/DailyDelhiClimateTest.csv")
delhi = train_series = pd.concat([dataTrain, dataTest])[["meantemp"]]
jena = pd.read_csv("./data/jena_climate_2009_2016.csv")[["T (degC)"]]
austin = pd.read_csv("./data/austin_weather.csv")[["TempAvgF"]]
bangladesh = pd.read_csv("./data/Temp_and_rain.csv")[["tem"]]
szeged = pd.read_csv("./data/weatherHistory.csv")[["Temperature (C)"]]

datasets = [
    (austin, "TempAvgF", "Austin"),
    (delhi, "meantemp", "Delhi"),
    (jena, "T (degC)", "Jena"),
    (bangladesh, "tem", "Bangladesh"),
    (szeged, "Temperature (C)", "Szeged")
]


def load_model():
    models = {}
    with open('saved_linreg.pkl', 'rb') as file:
        models["Linear Regression"] = pickle.load(file)

    with open('saved_mlp.pkl', 'rb') as file:
        models["MLP"] = pickle.load(file)

    with open('saved_lstm.pkl', 'rb') as file:
        models["LSTM"] = pickle.load(file)

    with open('saved_gru.pkl', 'rb') as file:
        models["GRU"] = pickle.load(file)

    with open('saved_rnn.pkl', 'rb') as file:
        models["RNN"] = pickle.load(file)
        
    if (not Path("./saved_arima.pkl").is_file()) and Path("./saved_arima.zip").is_file():
        with ZipFile("./saved_arima.zip", 'r') as zipObj:
            zipObj.extractall()

    with open('saved_arima.pkl', 'rb') as file:
        models['ARIMA'] = pickle.load(file)

    models['Transformer'] = {}
    for (data, key, city) in datasets:
        model = tf.keras.models.load_model(f'./Transformer+TimeEmbedding_{city}.hdf5',
                                           custom_objects={'Time2Vector': Time2Vector,
                                                           'SingleAttention': SingleAttention,
                                                           'MultiAttention': MultiAttention,
                                                           'TransformerEncoder': TransformerEncoder})

        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, train_size=0.77, shuffle=False)

        def generate_time_lags(df: pd.DataFrame, n_lags: int, col_name: str):
            df_n = df.copy()
            for n in range(1, n_lags + 1):
                df_n[f"lag{n}"] = df_n[col_name].shift(n)
            df_n = df_n.iloc[n_lags:]
            return df_n

        batch_size = 10
        seq_len = 20

        d_k = 40
        d_v = 40
        n_heads = 6
        ff_dim = 40

        lagged_train = generate_time_lags(train, LAG_FEATURES, key)
        train_data = lagged_train.values

        X_train, y_train = [], []
        for i in range(seq_len, len(train_data)):
            X_train.append(train_data[i - seq_len:i])  # Chunks of training data with a length of 128 df-rows
            y_train.append(train_data[:, 0][i])  # Value of 1st column (meantemp) of df-row 128+1
        X_train, y_train = np.array(X_train), np.array(y_train)

        models['Transformer'][city] = {
            'X': X_train,
            'key': key,
            'model': model
        }

    # with open('saved_transformer.pkl', 'rb') as file:
    #     models['Transformer'] = pickle.load(file)
    #     print(models['Transformer'])

    return models

loaded_Data = load_model()

def show_predict_page():
    st.title("Temperature Predictions")
    st.write("""### Need data to make predictions.""")

    cities = (
        "Delhi",
        "Bangladesh",
        "Szeged",
        "Austin",
        "Jena"
    )

    models = (
        "ARIMA",
        "GRU",
        "Linear Regression",
        "LSTM",
        "MLP",
        "RNN",
        "Transformer",
    )

    city = st.selectbox("City", cities)
    model = st.selectbox("Model", models)

    ok = st.button("Make prediction!")
    if ok:
        mirin = loaded_Data[model][city]
        X = mirin["X"]
        run_model = mirin["model"]

        pred = ...
        if model == "Linear Regression" or model == "MLP":
            pred = run_model.predict(X)
        if model == "LSTM" or model == "RNN" or model == "GRU":
            train_loader = torch.utils.data.DataLoader(X, batch_size=64, shuffle=False, drop_last=True)
            pred, values = run_model.evaluate(train_loader, batch_size=1, n_features=LAG_FEATURES)
            pred = pd.DataFrame([pred[i][0] for i in range(len(pred))])
        if model == "ARIMA":
            pred = run_model.predict()
        if model == 'Transformer':
            pred = run_model.predict(X)

        st.line_chart(pred)

