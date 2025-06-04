# Demo1234

This project demonstrates forecasting hotel occupancy using three approaches:
ARIMA, linear regression, and a neural network. The dataset spans 2024–2026 and
is stored in `hotel_factors_three_years.csv`.

## Usage

Install dependencies and run the predictor:

```bash
pip install -r requirements.txt  # or manually install pandas, scikit-learn, matplotlib, statsmodels
python hotel_predictor.py
```

The script prints mean absolute error for each method and saves a plot named
`occupancy_predictions.png` showing actual vs predicted occupancy for 2026.

## Tests

Tests verify that all models achieve reasonable accuracy. Run with:

```bash
pytest -v
```
