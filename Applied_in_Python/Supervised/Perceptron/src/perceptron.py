import numpy as np
import pandas as pd



class Perceptron:
    """Apply perceptron to split(classify) data using decision boundry."""
    def __init__(self, features, targets, epochs):
        self.data = features
        self.targets = targets
        self.epochs = epochs
        self.alpha = 0.01 # learning rate
        self.params = np.zeros(3)
        self.bias = 0

    def _predict_yhat(self, features):
        prediction = 0
        for theta, x in zip(self.params, features):
            prediction += theta * x
        prediction += self.bias
        return prediction

    def _calculate_cost(self, target, features):
        y_hat = self._predict_yhat(features)
        cost = target - y_hat
        return cost
        
    def _run_gradient(self):
        for num in range(self.epochs):
            for row, target in zip(self.data, self.targets):
                for i, x in enumerate(row):
                    cost = self._calculate_cost(target, row)
                    self.params[i] += self.alpha * cost * x
                self.bias += self.alpha * cost * x

    def make_prediction(self, features):
        self._run_gradient()
        y_hat = self._predict_yhat(features=features)
        return y_hat


# Load the data
data = pd.read_csv("data/data.csv")

# Scaling
def scale_data(features: pd.DataFrame, df_mean, df_std):
    new_features = (features - df_mean) / df_std
    return new_features

# Perceptron(features, targets, epochs)

targets = data["spam"]
data = data.drop("spam", axis=1)

df_mean = data.mean()
df_std = data.std()

scaled_data = scale_data(data, df_mean, df_std)

p = Perceptron(features=scaled_data[1:].to_numpy(), targets=targets[1:].to_numpy(), epochs=1500)

scaled_input = (np.array([10,8,105]) - df_mean.to_numpy()) / df_std.to_numpy()

prediction = p.make_prediction(scaled_input)

# print(paramas, bias)
print(p.params, p.bias)

# threshold for the perceptron is zero
if prediction < 0:
    print(0)
if prediction > 0:
    print(1)


# Model converged in less than an hour :) but it is just simple model so i think i should somehow complicate things.