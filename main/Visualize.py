import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, y_test, y_pred,title):
        self.y_test = y_test
        self.y_pred = y_pred
        self.title=title

    def plot_results(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.y_test, color='black', label='Test')
        plt.plot(self.y_pred, color='green', label='Predicted')
        plt.title(self.title)
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def scatter_plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue')
        plt.title('Scatter Plot of Test vs Predicted Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.show()

    def residual_plot(self):
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, residuals, color='red')
        plt.title('Residual Plot')
        plt.xlabel('Actual Prices')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.show()


