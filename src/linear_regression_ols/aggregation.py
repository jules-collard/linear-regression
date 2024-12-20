from linear_regression_ols import regression
import numpy as np

class Aggregator(regression.RegressionModel):
    """Class allowing simple aggregation of multiple regression models. Constructor takes a list of models (type :class:regression.RegressionModel) and averages their coefficients.

    :raises ValueError: Models must be fitted to identical data
    """

    def __init__(self, *models: regression.RegressionModel):
        """Constructor method
        """
        self.n = models[0].n
        self.p = models[0].p
        self.X = models[0].X
        self.y = models[0].y
        self.models = models
        self.fitted = False

        if not all(model.n == self.n and model.p == self.p for model in models):
            raise ValueError("Non-identically shaped design matrices")
        elif not all((model.X == self.X).all() and (model.y == self.y).all() for model in models):
            raise ValueError("Non-identical matrices")
        
        for model in models: model.check_fitted()
    
    def fit(self):
        """Fits model by aggregating coefficients.
        """
        self.beta_hat = sum([model.beta_hat for model in self.models]) / len(self.models)
        self.fitted = True
        
        self.y_hat = self.predict(self.X, add_intercept=False)
        self.residuals = self.y - self.y_hat