import numpy as np
import pandas as pd
import math


class LinearRegressionStochastic:
    def __init__(self, features, target, f_column: int):
        self.features = features
        self.target = target
        self.params = np.zeros(f_column) # initializing all weights + bias to zeros
        self.bias = 0
        self.alpha = 0.01 # Learning rate
        self.f_column = f_column

    def _predict_yhat(self, features):
        y_predict = 0
        for theta, x in zip(self.params, features):
            y_predict += theta * x
        y_predict += self.bias
        return y_predict

    def _calculate_cost(self, features, target_y):
        y_hat = self._predict_yhat(features)
        cost = (target_y - y_hat) ** 2
        return cost
    
    def _gradient_descent(self):
        for row, target in zip(self.features, self.target):
            for j in range(self.f_column):
                self.params[j] = self.params[j] - (self.alpha * (self._calculate_cost(row, target_y=target) * row[j]))
            self.bias = self.bias - (self.alpha * (self._calculate_cost(row, target_y=target)))
    
    def fit_data(self):
        self._gradient_descent()

    def show_predictions(self, size: float, bedrooms: int, age: int, distance: float):
        features = [size, bedrooms, age, distance]
        output = self._predict_yhat(features)
        return output


if __name__ == "__main__":
    data = pd.read_csv("Applied_in_Python/Supervised/data.csv")
    df_min = data.min()
    df_max = data.max()
    df_mean = data.mean()
    data = (data - data.mean()) / (data.max() - data.min())
    targets = data["House_Price"]
    data = data.drop("House_Price", axis=1)
    lr = LinearRegressionStochastic(np.array(data[1:]), np.array(targets[1:]), 4)

    def min_max(value, column):
        ret = (value - df_mean[column]) * (df_max[column] - df_min[column])
        return ret

    lr.fit_data()
    prediction = lr.show_predictions(size=min_max(1050, "Size_sqft"),
                                    bedrooms=min_max(4, "Bedrooms"), 
                                    age=min_max(3, "Age_years"), 
                                    distance=min_max(13, "Distance_city_km"))
    print(prediction * (df_max["House_Price"] - df_min["House_Price"]) + df_min["House_Price"])