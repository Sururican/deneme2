from sklearn.preprocessing import MinMaxScaler
from pandas_ta.utils import get_drift
from def_funktionen import *

class DataProcessor:
    def __init__(self, data, backcandles=30, test_size=0.2):
        self.data = data
        self.backcandles = backcandles
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train, self.X_test, self.y_test = self.prepare_data()

    def scale_data(self, data):
        return self.scaler.fit_transform(data)

    def prepare_X_y(self, scaled_data):
        X = []
        y = scaled_data[self.backcandles:, -1]
        y = np.reshape(y, (len(y), 1))

        for j in range(15):
            X.append([])
            for i in range(self.backcandles, scaled_data.shape[0]):
                X[j].append(scaled_data[i - self.backcandles:i, j])

        X = np.moveaxis(X, [0], [2])
        return X, y

    def split_data(self, X, y):
        split_limit = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_limit], X[split_limit:]
        y_train, y_test = y[:split_limit], y[split_limit:]
        return X_train, y_train, X_test, y_test

    def prepare_data(self):
        # Adding indicators
        
        self.data['RSI'] = ta.rsi(self.data['Close'], length=15)
        self.data['EMAF'] = ta.ema(self.data['Close'], length=20)
        self.data['EMAM'] = ta.ema(self.data['Close'], length=100)
        self.data['EMAS'] = ta.ema(self.data['Close'], length=150)
        self.data["CorrF"]=BCL2ECTS(self.data.High,self.data.Low,self.data.Open,self.data.Close,fast_length=20,slow_length=40,buy_threshold=0.5,sell_threshold=0)
        self.data["CorrF"]=[1 if self.data["CorrF"][i]>0 else 0 for i in range(len(self.data))]
        self.data["TII"]=TII(self.data.High,self.data.Low,self.data.Open,self.data.Close,majorLength=60,minorLength=30,upperLevel=80,lowerLevel=20)
        halftrend_out,halftrend_df=halftrend(self.data.High,self.data.Low,self.data.Open,self.data.Close)
        self.data["HalfTrend"]=halftrend_out.direction
        self.data["ASO"]=aso(self.data.High,self.data.Low,self.data.Open,self.data.Close,length=10)
        self.data["TPR"]=TPR(self.data.High,self.data.Low,self.data.Open,self.data.Close,length=14) 
        self.data["SuperTrend"]=ta.supertrend(self.data.High,self.data.Low,self.data.Close,length=7,multiplier=3,offset=0)["SUPERTd_7_3.0"]
        self.data["SuperTrend"]=[1 if self.data["SuperTrend"][i]>0 else 0 for i in range(len(self.data))]
        self.data['Target'] = self.data['SuperTrend'].shift(-1)
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True)
        self.data.drop(['Volume', 'Date'], axis=1, inplace=True)

        scaled_data = self.scale_data(self.data.iloc[:, 0:16])

        # Prepare X and y
        X, y = self.prepare_X_y(scaled_data)

        # Split data into train and test sets
        X_train, y_train, X_test, y_test = self.split_data(X, y)

        return X_train, y_train, X_test, y_test

