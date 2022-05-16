import pandas as pd
import streamlit as st
import pickle
import numpy as np
import sklearn
import torch
import torch.nn as nn
from classes.lstm import Optimization

LAG_FEATURES = 10

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
        # "ARIMA",
        "GRU",
        "Linear Regression",
        "LSTM",
        "MLP",
        "RNN",
        # "Transformer",
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

        st.line_chart(pred)

