from binance.client import Client
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Data:
    # Enter your Binance api key and api secret below
    client = Client(api_key, api_secret)

    def __init__(self, symbol, amount):
        self.symbol = symbol
        self.amount = amount

    def get_data(self):
        # Interval can be adjusted based on minute (m), hour (h), day (d) or week (w)
        price = self.client.get_klines(symbol=self.symbol, interval='1d', limit=self.amount)
        df = pd.DataFrame(price).drop([7, 9, 10, 11], axis=1)
        df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Num of Trades']
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        return df

    def data_visual(self):
        df = self.get_data()
        df['MA7'] = df['Close'].rolling(7).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        fig = go.Figure(data=[go.Candlestick(x=df['Close Time'], open=df['Open'], high=df['High'], low=df['Low'],
                                             close=df['Close'], name='OHLC'),
                              go.Scatter(x=df['Close Time'], y=df['MA7'], line=dict(color='purple', width=1.5),
                                         name='Moving Average (7 day)'),
                              go.Scatter(x=df['Close Time'], y=df['MA20'], line=dict(color='orange', width=1.5),
                                         name='Moving Average (20 day)')])
        fig.update_layout(title='Recent market data for: ' + self.symbol, yaxis_title='Price')
        fig.update_xaxes(rangeslider_visible=False)
        fig2 = px.line(df, x='Close Time', y='Close', title='Time Series for: ' + self.symbol)
        return fig.show(), fig2.show()

    def ml(self):
        df = self.get_data()
        ml_df = df.loc[:, ['Close', 'Volume', 'High']]
        ml_df['HL_pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
        ml_df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
        forecast_out = int(input('Enter number of days for forecast: '))
        ml_df['Prediction'] = df['Close'].shift(-forecast_out)
        X = np.array(ml_df.drop('Prediction', axis=1))
        X = X[:-forecast_out]
        y = np.array(ml_df['Prediction'])
        y = y[:-forecast_out]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


        models = [('LR', LinearRegression()), ('RDG', Ridge()), ('SVR', SVR())]
        x_forecast = np.array(ml_df.drop(['Prediction'], 1))[-forecast_out:]

        for name, model in models:
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            pred_future = model.predict(x_forecast)
            mse = mean_squared_error(y_test, predict)
            mae = mean_absolute_error(y_test, predict)
            r2 = r2_score(y_test, predict)
            print('---------------------------------------------------------')
            print('Mean Squared Error for %s is: %f' % (name, mse))
            print('Mean Absolute Error for %s is: %f' % (name, mae))
            # The closer the CoD is to 1, the stronger the correlation
            print('Coefficient of Determination for %s is: %f' % (name, r2))
            print('The next {} day(s) of stock predicted: {}'.format(forecast_out, pred_future))


def main():
    # Set cryptocurrency symbol and interval amount
    data = Data('ETHUSDT', int(input('Enter the number of days of data to receive (500 max): ')))
    # Get the tail (or most recent data) of the DataFrame
    print(data.get_data().tail())

    # Print machine learning function
    print(data.ml())

    # Visualize Data
    data.data_visual()


if __name__ == '__main__':
    main()
