
import tensorflow as tf
import numpy as np
from keras.models import load_model
from Financialdataprocessing import DataProcessor
import yfinance as yf
from datetime import datetime, timedelta


def make_predictions(input_data):
    # Load the saved model
    model_folder='Transformer_Modell'
    loaded_model = tf.keras.models.load_model(model_folder)

  
    predictions = loaded_model.predict(input_data)

    
    return predictions
end_date = datetime.now().strftime("%Y-%m-%d")
#start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
data = yf.download(tickers = 'EURUSD=X', start = '2000-03-11',end = end_date)
financial_data_processor = DataProcessor(data)
X_train, y_train, X_test, y_test = financial_data_processor.X_train, financial_data_processor.y_train, financial_data_processor.X_test, financial_data_processor.y_test
X=financial_data_processor.input_data
last_30_rows_array = X[-30:,:]
input=last_30_rows_array.reshape((1, 30, 11))

predictions=make_predictions(input)
print(f"Heutige Vorhersage : {predictions} Heute: {end_date}")


# %%
