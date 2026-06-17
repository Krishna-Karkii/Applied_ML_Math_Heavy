import numpy as np
import pandas as pd
import math

T = 0.4

class LocallyWeightedRegression:
    """Predict the Values Based on the closest values of the current variable we are making prediction for.
    """
    def __init__(self, 
                 features: np.array, 
                 targets: np.array,
                 df_mean,
                 df_std,
                 epoch=700):
        self.features = features
        self.targets = targets
        self.params = np.zeros(3, dtype=np.float32)
        self.bias = 0
        self.mean = df_mean
        self.std = df_std
        self.epoch = epoch
        self.alpha = 0.01 # Learning rate

    def predict_yhat(self, x_preds):
        self._fit_line(x_preds)
        prediction = 0
        for i, x in enumerate(self.params):
            prediction += x * x_preds[i]
        prediction += self.bias
        prediction = self._rescale_output(prediction)
        return prediction

    def _calculate_cost(self, features, target):
        """Calculating the cost of the current parameters."""
        cost = 0
        for i, x in enumerate(features):
            cost += x * self.params[i]
        cost += self.bias
        cost = (cost - target)
        return cost


    def _calculate_weight(self, x_pred, x):
        """Calculating the absolute value of x(current num)"""
        new_val = math.e ** (-1 * ((x - x_pred)**2)/ (2 * (T ** 2)))
        return new_val
    
    def _rescale_output(self, y_hat):
        rescaled = (y_hat * self.std) + self.mean
        return rescaled
    
    def _fit_line(self, x_preds):
        for a in range(self.epoch):
            for i, row in enumerate(self.features):
                for j, x in enumerate(row):
                    cost = self._calculate_cost(row, self.targets[i])
                    self.params[j] -= self.alpha * cost * self._calculate_weight(x_preds[j], x) * x
                    self.bias -= cost * self.alpha

def rescale_data(data: pd.DataFrame):
    """Rescaling the whole data from -1 to 1 to avoid overshooting values."""
    rescaled_data = (data - data.mean()) / data.std()
    return rescaled_data

def rescale_input(data: np.array, values: np.array):
    rescaled_data = ((values - data.mean()) / data.std())
    return rescaled_data
    

if __name__ == "__main__":
    data = pd.read_csv("data/housing_price.csv")
    df_mean = data.mean()["price_k"]
    df_std = data.std()["price_k"]
    re_data = rescale_data(data=data)
    targets = re_data["price_k"]
    re_data = re_data.drop("price_k", axis=1)
    lwr = LocallyWeightedRegression(features=re_data[1:].to_numpy(), 
                                    targets=targets[1:].to_numpy(),
                                    df_mean=df_mean,
                                    df_std=df_std)
    output = lwr.predict_yhat(rescale_input(data=data[1:].to_numpy(), values=np.array([1200,3,6])))
    print(lwr.params, lwr.bias)
    print(output * 1000)