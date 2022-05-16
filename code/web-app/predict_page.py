import streamlit as st
import pickle
import numpy as np
import sklearn
import torch
import torch.nn as nn

def load_model():
    models = {}
    with open('saved_linreg.pkl', 'rb') as file:
        models["Linear Regression"] = pickle.load(file)

    with open('saved_mlp.pkl', 'rb') as file:
        models["MLP"] = pickle.load(file)

    with open('saved_lstm.pkl', 'rb') as file:
        models["LSTM"] = pickle.load(file)

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
        #"GRU",
        "Linear Regression",
        "LSTM",
        "MLP",
        # "RNN",
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
        if model == "LSTM":
            output_dim = 1
            hidden_dim = 64
            layer_dim = 3
            batch_size = 64
            dropout = 0.2
            n_epochs = 100
            learning_rate = 1e-3
            weight_decay = 1e-6

            train_loader = torch.utils.data.DataLoader(X, batch_size=64, shuffle=False, drop_last=True)
            loss_fn = nn.MSELoss(reduction="mean")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            from classes.lstm import Optimization
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
            opt.train(train_loader, test_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
            pred, values = run_model.evaluate(train_loader, batch_size=1, n_features=X.columns)
            print(pred)

        return
        st.line_chart(pred)

