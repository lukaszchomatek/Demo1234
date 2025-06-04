import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = 'hotel_factors_three_years.csv'
FEATURES = [
    'Day_of_Year',
    'Domestic_Guests',
    'Foreign_Guests',
    'Flight_Price',
    'Reviews',
    'Event_Flag',
    'Season_Flag',
    'Availability',
    'Hotel_Price',
    'Competing_Hotels_Availability'
]


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    train = df[df['Year'] < 2026].reset_index(drop=True)
    test = df[df['Year'] == 2026].reset_index(drop=True)
    return train, test


def predict_arima(train: pd.DataFrame, test: pd.DataFrame):
    model = ARIMA(train['Occupancy'], order=(1, 0, 1)).fit()
    forecast = model.forecast(steps=len(test))
    mae = mean_absolute_error(test['Occupancy'], forecast)
    return forecast, mae


def predict_regression(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[FEATURES], train['Occupancy']
    X_test, y_test = test[FEATURES], test['Occupancy']
    model = LinearRegression().fit(X_train, y_train)
    forecast = model.predict(X_test)
    mae = mean_absolute_error(y_test, forecast)
    return forecast, mae


def predict_nn(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[FEATURES], train['Occupancy']
    X_test, y_test = test[FEATURES], test['Occupancy']
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
    ])
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    mae = mean_absolute_error(y_test, forecast)
    return forecast, mae


def plot_results(test: pd.DataFrame, preds: dict):
    days = range(len(test))
    plt.figure(figsize=(10, 6))
    plt.plot(days, test['Occupancy'], label='Actual')
    for name, values in preds.items():
        plt.plot(days, values, label=name)
    plt.xlabel('Day of 2026')
    plt.ylabel('Occupancy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('occupancy_predictions.png')


def main():
    df = load_data()
    train, test = split_data(df)
    arima_pred, arima_mae = predict_arima(train, test)
    reg_pred, reg_mae = predict_regression(train, test)
    nn_pred, nn_mae = predict_nn(train, test)

    print(f'ARIMA MAE: {arima_mae:.2f}')
    print(f'Regression MAE: {reg_mae:.2f}')
    print(f'Neural Net MAE: {nn_mae:.2f}')

    plot_results(test, {
        'ARIMA': arima_pred,
        'Regression': reg_pred,
        'NeuralNet': nn_pred
    })

    return {
        'arima_mae': arima_mae,
        'regression_mae': reg_mae,
        'nn_mae': nn_mae,
        'predictions': {
            'arima': arima_pred,
            'regression': reg_pred,
            'nn': nn_pred
        }
    }


if __name__ == '__main__':
    main()
