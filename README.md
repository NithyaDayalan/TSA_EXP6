## Ex.No: 6 HOLT WINTERS METHOD
### DATE: 15/04/2025
### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
```
Devloped by: NITHYA D
Register Number: 212223240110
```

#### Import necessary libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
```
#### Load the dataset
```
data = pd.read_csv('Gold Price Prediction.csv', parse_dates=['Date'], index_col='Date')
``` 
#### Use 'Price Today' for modeling
```
data_monthly = data['Price Today'].resample('MS').mean()
```
#### Plot original monthly data
```
data_monthly.plot(title='Monthly Gold Prices - Original')
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.grid(True)
plt.show()
```
#### Scale the data
```
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
```
#### Plot scaled data
```
scaled_data.plot(title='Scaled Monthly Gold Prices')
plt.xlabel("Date")
plt.ylabel("Scaled Value")
plt.grid(True)
plt.show()
```
#### Decompose the original (non-scaled) data to visualize trend and seasonality
```
decomposition = seasonal_decompose(data_monthly, model='additive')
decomposition.plot()
plt.show()
```
#### Shift scaled data to make all values positive (needed for multiplicative seasonality)
```
scaled_data += 1
```
#### Split into training and testing sets (80/20)
```
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
```
#### Train Holt-Winters model with additive trend and multiplicative seasonality
```
model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()
```
#### Forecast on test data
```
test_predictions = model.forecast(len(test_data))
```
#### Plot predictions vs actual
```
ax = train_data.plot(label='Train')
test_data.plot(ax=ax, label='Test')
test_predictions.plot(ax=ax, label='Forecast')
ax.set_title('Gold Price Forecast - Visual Evaluation')
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Value')
ax.legend()
plt.grid(True)
plt.show()
```
#### Evaluate forecast
```
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print(f"RMSE on test data: {rmse:.4f}")
print("Scaled Std Dev:", np.sqrt(scaled_data.var()))
print("Scaled Mean:", scaled_data.mean())
```
#### Train final model on full dataset
```
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()
```
#### Forecast future gold prices (next 8 months, adjust as needed)
```
future_steps = 8
final_forecast_scaled = final_model.forecast(steps=future_steps)
```
#### Inverse scaling to get original price scale
```
final_forecast = pd.Series(
    scaler.inverse_transform((final_forecast_scaled - 1).values.reshape(-1, 1)).flatten(),
    index=pd.date_range(start=scaled_data.index[-1] + pd.offsets.MonthBegin(1), periods=future_steps, freq='MS')
)
```
#### Plot historical and forecasted prices
```
ax = data_monthly.plot(label='Historical')
final_forecast.plot(ax=ax, label='Forecast', style='--')
ax.set_title('Future Gold Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price (USD)')
ax.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:
#### Original_data plot:
![image](https://github.com/user-attachments/assets/e48d9df0-1d34-467e-9f00-acd3acb82c13)

#### Scaled_data plot:
![image](https://github.com/user-attachments/assets/1bc2dd2c-839b-401f-bae8-499ba749feff)

#### Decomposed plot:
![image](https://github.com/user-attachments/assets/9db84bda-bf99-4267-861a-dbd61dae2973)

#### Test prediction:
![image](https://github.com/user-attachments/assets/f200d89e-9fa8-4815-9bf6-4218b7254010)

#### Model performance metrics :
##### RMSE:
![image](https://github.com/user-attachments/assets/5131d43f-0d45-4f55-a324-8429075cf28c)
##### Standard deviation:
![image](https://github.com/user-attachments/assets/a3cff0fb-5b7f-4714-ae7d-02187be86204)
##### Mean:
![image](https://github.com/user-attachments/assets/aa7315a3-f34b-4604-9709-fe527528b1e3)

#### Final prediction:
![image](https://github.com/user-attachments/assets/d50df0c6-c058-4f6b-8166-1211078bcca0)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
