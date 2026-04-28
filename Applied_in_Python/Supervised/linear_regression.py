import numpy as np

class LinearRegressionStochastic:
    def __init__(self, features: np.array, target: np.array, f_row: int, f_column: int):
        self.features = features
        self.target = target
        self.params = np.zeros(f_row) # initializing all weights + bias to zeros
        self.bias = np.zeros(f_column)
        self.alpha = 0.02 # Learning rate
        self.f_column = f_column

    def _predict_yhat(self, features: np.array, col):
        count = 0
        for theta in self.params:
            if count == 0:

    def _calculate_cost(self, cur_features: np.array, target_y):
        y_hat = self._predict_yhat(cur_features, col)
        cost = (y_hat - target_y) ** 2
        return cost
        
    def _gradient_descent(self):
        for row, target in zip(self.features, self.target):
            for j in range(self.f_column):
                self.params[j] = self.params[j] - self.alpha * (self._calculate_cost(row, target_y=target) * row[j])
    def show_predictions(self):
        pass


if __name__ == "__main__":
    x = np.array()
    y = np.array()
    lr = LinearRegressionStochastic()
    lr.show_predictions()