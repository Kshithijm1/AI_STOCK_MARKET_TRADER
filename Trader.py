import backtrader as bt
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load historical stock price data (replace 'AAPL.csv' with your dataset)
data = pd.read_csv('AAPL.csv')
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Feature Engineering
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Calculate RSI
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
short_ema = data['Close'].ewm(span=12, adjust=False).mean()
long_ema = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = short_ema - long_ema

# Normalize the data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']] = scaler.fit_transform(
    data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']])

# Create a sequence of historical data for training
sequence_length = 10  # Adjust this for your needs
sequences = []
next_prices = []

for i in range(len(data) - sequence_length):
    sequences.append(data.iloc[i:i+sequence_length].values)
    next_prices.append(data.iloc[i+sequence_length]['Close'])

X = np.array(sequences)
y = np.array(next_prices)

# Split the data into training and testing sets
split_ratio = 0.8  # Adjust as needed
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create an LSTM model for price prediction
model = Sequential([
    LSTM(units=100, activation='relu', input_shape=(sequence_length, X.shape[2])),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Create a custom strategy class for trading
class SmarterTradingStrategy(bt.Strategy):
    params = (
        ("prediction_interval", 5),  # Prediction interval in days
        ("risk_percent", 2.0),  # Maximum risk per trade (2% of portfolio)
        ("stop_loss_percent", 1.0),  # Dynamic stop loss (1% of volatility)
        ("take_profit_percent", 2.0),  # Dynamic take profit (2% of volatility)
    )

    def __init__(self):
        self.data_close = self.data.close
        self.buy_signal_triggered = False
        self.buy_price = None
        self.volatility = bt.indicators.AverageTrueRange(period=14)

    def next(self):
        current_price = self.data_close[0]

        # Make a prediction for the next `prediction_interval` days
        prediction_data = np.array([self.data.values[-sequence_length:]])
        predicted_price = model.predict(prediction_data)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        if current_price < predicted_price and not self.position:
            # Calculate position size based on risk
            risk_percent = self.params.risk_percent / 100
            risk_amount = self.broker.getvalue() * risk_percent

            # Calculate dynamic stop loss and take profit
            stop_loss = self.params.stop_loss_percent / 100 * self.volatility[0]
            take_profit = self.params.take_profit_percent / 100 * self.volatility[0]

            size = int(risk_amount / (current_price * (1 - stop_loss)))

            # Buy signal
            self.buy(size=size)
            self.buy_signal_triggered = True
            self.buy_price = current_price
            print(f"Buy Signal at {self.data.datetime.datetime()}")

        elif current_price > predicted_price and self.position:
            # Sell signal
            self.sell()
            self.buy_signal_triggered = False
            print(f"Sell Signal at {self.data.datetime.datetime()}")

# Create a backtest engine
cerebro = bt.Cerebro()

# Add data feed
data_feed = bt.feeds.PandasData(dataname=data)

cerebro.adddata(data_feed)

# Add the trading strategy
cerebro.addstrategy(SmarterTradingStrategy)

# Set cash and commission
initial_cash = 100000  # Adjust as needed
cerebro.broker.set_cash(initial_cash)
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission per trade

# Print the starting cash
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

# Run the backtest
cerebro.run()

# Print the final portfolio value
print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

# Estimate the price after a given amount of time
prediction_interval = 30  # Change this to your desired prediction interval (in days)

# Prepare data for prediction
last_sequence = np.array([data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']][-sequence_length:].values])
last_sequence = last_sequence.reshape(1, sequence_length, X.shape[2])

# Predict the price after the specified interval
predicted_price = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

# Print the estimated price
print(f"Estimated Price after {prediction_interval} days: {predicted_price:.2f}")
