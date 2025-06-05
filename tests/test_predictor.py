import pandas as pd
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hotel_predictor import load_data, split_data, predict_arima, predict_regression, predict_nn

class TestPredictor(unittest.TestCase):
    def test_arima_mae(self):
        df = load_data()
        train, test = split_data(df)
        _, mae = predict_arima(train, test) # Default order is (0,0,1)
        print(f"ARIMA MAE (0,0,1): {mae}")
        self.assertLess(mae, 4.0)

    def test_regression_mae(self):
        df = load_data()
        train, test = split_data(df)
        _, mae = predict_regression(train, test)
        print(f"Regression MAE: {mae}")
        self.assertLess(mae, 1.3)

    def test_nn_mae(self):
        df = load_data()
        train, test = split_data(df)
        _, mae = predict_nn(train, test) # Default HLS=(100,50), MI=1000
        print(f"NN MAE: {mae}")
        self.assertLess(mae, 1.4)

if __name__ == '__main__':
    unittest.main()
