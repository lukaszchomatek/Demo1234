import pandas as pd
from hotel_predictor import load_data, split_data, predict_arima, predict_regression, predict_nn

def test_arima_mae():
    df = load_data()
    train, test = split_data(df)
    _, mae = predict_arima(train, test)
    assert mae < 5

def test_regression_mae():
    df = load_data()
    train, test = split_data(df)
    _, mae = predict_regression(train, test)
    assert mae < 2

def test_nn_mae():
    df = load_data()
    train, test = split_data(df)
    _, mae = predict_nn(train, test)
    assert mae < 5
