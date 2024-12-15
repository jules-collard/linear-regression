import regression
import numpy as np

class Aggregator(regression.RegressionModel):

    def __init__(self, *models: regression.RegressionModel):
        self.n = models[0].n
        self.p = models[0].p
        self.X = models[0].X
        self.y = models[0].y
        self.models = models
        self.fitted = False

        if not all(model.n == self.n and model.p == self.p for model in models):
            raise ValueError("Non-identically shaped deisgn matrices")
        elif not all((model.X == self.X).all() and (model.y == self.y).all() for model in models):
            raise ValueError("Non-identical matrices")
        
        for model in models: model.check_fitted()
    
    def fit(self):
        self.beta_hat = sum([model.beta_hat for model in self.models]) / len(self.models)
        self.fitted = True
        
        self.y_hat = self.predict(self.X, add_intercept=False)
        self.residuals = self.y - self.y_hat

def test():
    X = np.random.rand(100, 2)  
    y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) * 0.5  # linear model with gaussian noise

    mod1 = regression.OLSModel(X,y)
    mod1.fit()
    mod2 = regression.WLSModel(X,y)
    mod2.fit()
    mod3 = regression.RidgeModel(X,y)
    mod3.fit(5)

    agg = Aggregator(mod1, mod2, mod3)
    agg.fit()
    agg.summary()


if __name__ == "__main__":
    test()