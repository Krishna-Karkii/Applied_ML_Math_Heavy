import numpy as np
import pandas as pd
import math


class LogisticRegression:
    """Logistic regression to predict if we can provide loan to the current customer
    Based on the income credit score and debt."""
    def __init__(self, rows, targets, epochs=3000):
        self.data = rows
        self.targets = targets
        self.epochs = epochs
        self.rows = len(rows)
        self.params = np.zeros(3)
        self.bias = 0
        self.alpha = 0.01 # Learning rate


    def _predict_yhat(self, features):
        """predict the y_hat value for the corresponding features"""
        # Calculate theta transpose x
        value = 0
        for theta, x in zip(self.params, features):
            value += theta * x
        value += self.bias

        # Calculate the prediction value.
        prediction = 1 / (1 + math.e ** (-1 * value))
        return prediction


    def _calculate_cost(self, target, features):
        y_hat = self._predict_yhat(features)
        cost = (target - y_hat)
        return cost
        
    def _gradient_ascent(self):
        for e in range(self.epochs):
            for row, target in zip(self.data, self.targets):
                for i, x in enumerate(row):
                    cost = self._calculate_cost(target=target, features=row)
                    self.params[i] += self.alpha * cost * x
                    self.bias += self.alpha * cost
    
    def make_prediction(self, features):
        self._gradient_ascent()
        prediction = self._predict_yhat(features)
        return prediction

def scaler(data: pd.DataFrame):
    data = (data - data.mean())/ data.std()
    return data

data = pd.read_csv("data/data.csv")
targets = data["approved"]
data = data.drop("approved", axis=1)
df_mean = data.mean().to_numpy()
df_std = data.std().to_numpy()
to_predict = np.array([60,670,24])
data = scaler(data=data)

lr = LogisticRegression(data[1:].to_numpy(), targets[1:].to_numpy())

to_predict = (to_predict - df_mean)/ df_std
out = lr.make_prediction(to_predict)

if out < 0.5:
    print(0)
else:
    print(1)



print(lr.params)
# Can't beleive i made this in one go hahahahahaha and it seems to work