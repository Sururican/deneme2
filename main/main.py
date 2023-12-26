
from LSTMBayesianOptimizer import BayesianLSTMOptimizer, LSTMModel
from Evaluate import Evaluate
from Financialdataprocessing import DataProcessor
from Visualize import Visualizer
import yfinance as yf
from transformer import TransformerModel, BayesianTransformerOptimizer
data = yf.download(tickers = '^RUI', start = '2000-03-11',end = '2023-10-10')
financial_data_processor = DataProcessor(data)
X_train, y_train, X_test, y_test = financial_data_processor.X_train, financial_data_processor.y_train, financial_data_processor.X_test, financial_data_processor.y_test

bayesian_optimizer = BayesianLSTMOptimizer(X_train, y_train, 30)
bayesian_optimizer.optimize()

best_hps = bayesian_optimizer.best_hps

best_hps.values

lstm=LSTMModel((30,15),best_hps.values['units'],best_hps.values['learning_rate'],best_hps.values['activation'])
lstm.train_model(X_train, y_train)
lstmmodel=lstm.model
y_pred=lstmmodel.predict(X_test)
for i in range(10):
    print(y_pred[i], y_test[i])
import numpy as np
visualizer = Visualizer(y_test, np.round(y_pred),'LSTM TEST vs Predictive Results')
visualizer.plot_results()

transformer_optimizer=BayesianTransformerOptimizer(X_train,y_train)
transformer_optimizer.optimize()

values=transformer_optimizer.best_hps.values
values

transformer_model = TransformerModel(head_size=values['head_size'], num_heads=values['num_heads'], ff_dim=values['ff_dim'], num_trans_blocks=4, mlp_units=[values['mlp_units']], mlp_dropout=values['mlp_dropout'], dropout=values['dropout'], attention_axes=1)

transformer_model.compile_and_fit(X_train, y_train)
y_pred_Trans=transformer_model.model.predict(X_test)
for i in range(10):
    print(y_pred_Trans[i], y_test[i])
    
visualizer2 = Visualizer(y_test, np.round(y_pred_Trans),'Transformer TEST vs Predictive Results')
visualizer2.plot_results()
evaluate=Evaluate(y_test,y_pred_Trans)
print(evaluate.compare_var)
print(evaluate.evaluate_model_with_mape)




