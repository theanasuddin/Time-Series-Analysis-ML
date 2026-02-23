import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from scipy import stats

df = pd.read_csv('Download Data - STOCK_US_XNAS_AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
df['Close'] = df['Close'].astype(float)

stock_data = df['Close'].values
split_idx = int(len(stock_data) * 0.8)
train_data = stock_data[:split_idx]
test_data = stock_data[split_idx:]

# Stationarity test
print("\nDifferencing order test:")
result = adfuller(train_data)
print(f'Original p-value: {result[1]:.6f}')
result = adfuller(np.diff(train_data))
print(f'1st diff p-value: {result[1]:.6f}')

# ACF/PACF plots
d_order = 1
diff_data = np.diff(train_data)
z_score = 3
reference = z_score / np.sqrt(len(diff_data))

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(diff_data, lags=40, ax=ax1)
ax1.set_title('ACF (for MA order q)')
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(diff_data, lags=40, ax=ax2)
ax2.axhline(y=reference, color='r', linestyle='--')
ax2.axhline(y=-reference, color='r', linestyle='--')
ax2.set_title('PACF (for AR order p)')
plt.tight_layout()
plt.show()

print("\nTesting ARIMA and SARIMA configurations:")
arima_configs = [
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 1),
    (4, 1, 1),
    (5, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 2),
    (4, 1, 2),
    (1, 1, 3),
    (2, 1, 3),
    (3, 1, 3),
]

sarima_configs = [
    ((1, 1, 1), (1, 0, 1, 5)),
    ((2, 1, 1), (0, 1, 1, 5)),
    ((1, 1, 2), (1, 1, 0, 5)),
]

results_summary = []
best_rmse = np.inf
best_config = None
best_model = None
best_type = None
best_predictions = None

# Test ARIMA configs
for config in arima_configs:
    try:
        model = ARIMA(train_data, order=config).fit()
        preds = model.forecast(steps=len(test_data))
        rmse = np.sqrt(np.mean((test_data - preds)**2))
        results_summary.append({
            'type': 'ARIMA',
            'config': config,
            'AIC': model.aic,
            'BIC': model.bic,
            'RMSE': rmse
        })
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config
            best_model = model
            best_type = 'ARIMA'
            best_predictions = preds
        print(f"ARIMA{config}: RMSE={rmse:.4f}, AIC={model.aic:.2f}")
    except Exception as e:
        print(f"ARIMA{config} failed: {e}")

# Test SARIMA configs
for order, seasonal_order in sarima_configs:
    try:
        model = SARIMAX(train_data, order=order,
                        seasonal_order=seasonal_order).fit(disp=False)
        preds = model.forecast(steps=len(test_data))
        rmse = np.sqrt(np.mean((test_data - preds)**2))
        results_summary.append({
            'type': 'SARIMA',
            'config': (order, seasonal_order),
            'AIC': model.aic,
            'BIC': model.bic,
            'RMSE': rmse
        })
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = (order, seasonal_order)
            best_model = model
            best_type = 'SARIMA'
            best_predictions = preds
        print(
            f"SARIMA{order}x{seasonal_order}: RMSE={rmse:.4f}, AIC={model.aic:.2f}"
        )
    except Exception as e:
        print(f"SARIMA{order}x{seasonal_order} failed: {e}")

print(f"\nBest model: {best_type}{best_config}, RMSE={best_rmse:.4f}")

# Residual analysis
dw_stat = sm.stats.durbin_watson(best_model.resid)
print(f"\nDurbin-Watson: {dw_stat:.4f}")

fig = plt.figure(figsize=(12, 6))
plt.plot(best_model.resid)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title(f'Residuals {best_type}{best_config}')
plt.xlabel('Sample')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

norm_test = stats.normaltest(best_model.resid)
print(f"Normality test p-value: {norm_test.pvalue:.4f}")

fig = plt.figure(figsize=(12, 6))
qqplot(best_model.resid, line="q", fit=True)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(best_model.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(best_model.resid, lags=40, ax=ax2)
plt.tight_layout()
plt.show()

# Forecast evaluation
mae = np.mean(np.abs(test_data - best_predictions))
rmse = np.sqrt(np.mean((test_data - best_predictions)**2))
mape = np.mean(np.abs((test_data - best_predictions) / test_data)) * 100
print(f"\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# Plot predictions
fig = plt.figure(figsize=(14, 7))
x_train = np.arange(len(train_data))
x_test = np.arange(len(train_data), len(train_data) + len(test_data))

plt.plot(x_train, train_data, 'b-', label='Training', alpha=0.7)
plt.plot(x_test, test_data, 'g-', label='Actual', linewidth=2)
plt.plot(x_test, best_predictions, 'r--', label='Predicted', linewidth=2)
plt.axvline(x=len(train_data), color='k', linestyle=':', alpha=0.5)
plt.xlabel('Time (days)')
plt.ylabel('Price ($)')
plt.title(f'AAPL {best_type}{best_config} Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(test_data,
         'g-',
         label='Actual',
         linewidth=2,
         marker='o',
         markersize=4)
plt.plot(best_predictions,
         'r--',
         label='Predicted',
         linewidth=2,
         marker='s',
         markersize=4)
plt.xlabel('Test Sample')
plt.ylabel('Price ($)')
plt.title(f'{best_type}{best_config} Test Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Top 5 models by RMSE
results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('RMSE').head(5)
print("\nTop 5 configurations by RMSE:")
print(results_df.to_string(index=False))
