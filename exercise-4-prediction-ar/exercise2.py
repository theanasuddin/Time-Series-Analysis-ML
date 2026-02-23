import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.api import qqplot

df = pd.read_pickle('Nepal_electricity_consumption_in_MWh.pkl')
df = df.sort_values('Date').reset_index(drop=True)

consumption_col = [
    c for c in df.columns if 'consumption' in c.lower() or 'mwh' in c.lower()
]
if consumption_col:
    electricity_data = df[consumption_col[0]].values
else:
    electricity_data = df.iloc[:, 1].values

electricity_data = pd.Series(electricity_data).ffill().bfill().values
electricity_data = electricity_data[np.isfinite(electricity_data)]

# Train/test split (80/20)
split_idx = int(len(electricity_data) * 0.8)
train_data = electricity_data[:split_idx]
test_data = electricity_data[split_idx:]

# Plot 400-500 samples
plt.figure(figsize=(12, 5))
plt.plot(electricity_data[:500], linewidth=1)
plt.title('Electricity Consumption (First 500 samples)')
plt.xlabel('Time')
plt.ylabel('Consumption (MWh)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Stationarity test
result = adfuller(train_data)
print(f'ADF p-value (original): {result[1]:.6f}')
result = adfuller(np.diff(train_data))
print(f'ADF p-value (1st diff): {result[1]:.6f}')

# Try ARIMA/SARIMA configs
configs = [(1, 1, 1, None), (2, 1, 1, None), (3, 1, 1, None), (1, 1, 2, None),
           (2, 1, 2, None), (1, 1, 1, 7), (2, 1, 1, 7), (1, 1, 1, 12),
           (2, 1, 1, 12)]

best_rmse = np.inf
best_config = None
results_summary = []

for config in configs:
    try:
        p, d, q, s = config
        if s is None:
            model = ARIMA(train_data, order=(p, d, q)).fit()
        else:
            model = ARIMA(train_data,
                          order=(p, d, q),
                          seasonal_order=(1, 1, 1, s)).fit()
        predictions = model.forecast(steps=len(test_data))
        rmse = np.sqrt(np.mean((test_data - predictions)**2))
        results_summary.append({'config': config, 'RMSE': rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config
    except:
        continue

print(f"\nBest config (by RMSE): ARIMA{best_config}, RMSE={best_rmse:.4f}")

# Fit final model
p, d, q, s = best_config
if s is None:
    best_model = ARIMA(train_data, order=(p, d, q)).fit()
    config_str = f"({p},{d},{q})"
else:
    best_model = ARIMA(train_data,
                       order=(p, d, q),
                       seasonal_order=(1, 1, 1, s)).fit()
    config_str = f"({p},{d},{q})x(1,1,1,{s})"

# Residual analysis
dw_stat = sm.stats.durbin_watson(best_model.resid)
print(f"Durbin-Watson: {dw_stat:.4f}")
norm_test = stats.normaltest(best_model.resid)
print(f"Residual normality p-value: {norm_test.pvalue:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(best_model.resid)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title(f'Residuals ARIMA{config_str}')
plt.tight_layout()
plt.show()

qqplot(best_model.resid, line="q", fit=True)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()

# Forecast and evaluation
predictions = best_model.forecast(steps=len(test_data))
mae = np.mean(np.abs(test_data - predictions))
rmse = np.sqrt(np.mean((test_data - predictions)**2))
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# Plot predictions
x_train = np.arange(len(train_data))
x_test = np.arange(len(train_data), len(train_data) + len(test_data))

plt.figure(figsize=(13, 6))
plt.plot(x_train, train_data, 'b-', label='Training')
plt.plot(x_test, test_data, 'g-', label='Actual', linewidth=2)
plt.plot(x_test, predictions, 'r--', label='Predicted', linewidth=2)
plt.axvline(x=len(train_data), color='k', linestyle=':', alpha=0.5)
plt.title(f'Prediction ARIMA{config_str}')
plt.xlabel('Time')
plt.ylabel('Consumption (MWh)')
plt.legend()
plt.tight_layout()
plt.show()

# Show compact summary of top results
results_df = pd.DataFrame(results_summary).sort_values('RMSE').head(5)
print("\nTop configurations (by RMSE):")
print(results_df.to_string(index=False))
