import streamlit
from classes.lstm import LSTMModel, Optimization
from classes.gru import GRUModel
from classes.rnn import RNNModel
from rmse import RMSELoss
from statsmodels.tsa.arima.model import ARIMA
from predict_page import show_predict_page

show_predict_page()