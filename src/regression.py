import numpy as np
from scipy.stats import t, f
from abc import ABC, abstractmethod

class RegressionModel(ABC):

    def __init__(self, X: np.ndarray, y: np.ndarray, intercept=True):
        self.fitted = False
        # Data
        self.X: np.ndarray = np.hstack((np.ones((X.shape[0], 1)), X)) if intercept else X
        self.y: np.ndarray = y
        self.n, self.p = self.X.shape

        # Model Output
        self.beta_hat: np.ndarray = None
        self.residuals: np.ndarray = None
        self.y_hat: np.ndarray = None

        # Diagnostics
        self.aic: float = None
        self.bic: float = None
        self.r2: float = None
        self.adj_r2: float = None

    # Fits model to data
    @abstractmethod
    def fit(self):
        pass

    def predict(self, X_new):
        self.check_fitted()
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
        return X_new @ self.beta_hat
    
    # TO IMPLEMENT
    @abstractmethod
    def summary(self):
        pass

    def check_fitted(self):
        if not self.fitted: raise ValueError('Model not fitted')
        

#implementation uses numpy for basic matrix operations.
#Also uses scipy to determine the quantiles of the t and f distribution (could be done from scratch but is not the focus of the project, imo)
class OLSModel(RegressionModel):
    
    def __init__(self, X, y, intercept=True):
        RegressionModel.__init__(self, X, y, intercept=intercept)
        self.sigma_squared: float = None
        self.var_beta = None #estimated variance of betahat

    # Calculate OLS estimator and resulting residuals
    def fit(self):
        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.beta_hat = XtX_inv @ self.X.T @ self.y
        self.y_hat = self.X @ self.beta_hat
        self.residuals = self.y - self.y_hat
        self.sigma_squared = (self.residuals.T @ self.residuals) / (self.n - self.p)
        self.var_beta = self.sigma_squared * XtX_inv
        self.fitted = True

    def get_covariance_matrix(self):
        self.check_fitted()
        return self.var_beta


class Tester:
    def __init__(self, ols: OLSModel):
        ols.check_fitted()
        self.ols = ols

    def confidence_intervals(self, alpha=0.05):
        var_beta = self.ols.get_covariance_matrix()
        se_beta = np.sqrt(np.diag(var_beta))
        t_critical = t.ppf(1 - alpha / 2, self.ols.n - self.ols.p)
        lower_bounds = self.ols.beta_hat - t_critical * se_beta
        upper_bounds = self.ols.beta_hat + t_critical * se_beta
        return np.column_stack((lower_bounds, upper_bounds))

    def t_statistics(self, hypothesized_values=None): 
        if hypothesized_values is None:
            hypothesized_values = np.zeros_like(self.ols.beta_hat) # default: b_j = 0, for all j
        var_beta = self.ols.get_covariance_matrix()
        se_beta = np.sqrt(np.diag(var_beta))
        return (self.ols.beta_hat - hypothesized_values) / se_beta


    def t_test(self, hypothesized_values=None, alpha=0.05):
        t_stats = self.t_statistics(hypothesized_values=hypothesized_values)
       
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=self.ols.n - self.ols.p))
        # Evaluate significance
        significant = p_values < alpha
        return t_stats, p_values, significant


    def f_test(self, R, r):
        # F-test for H0: Rβ = r
        R_beta_hat = R @ self.ols.beta_hat - r
        cov_R_beta = R @ self.ols.get_covariance_matrix() @ R.T
        F_stat = (R_beta_hat.T @ np.linalg.inv(cov_R_beta) @ R_beta_hat) / R.shape[0]
        F_stat /= self.ols.sigma_squared
        p_value = 1 - f.cdf(F_stat, R.shape[0], self.ols.n - self.ols.p)
        return F_stat, p_value

    def prediction_intervals(self, X_new, alpha=0.05):
        predictions = self.ols.predict(X_new)
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))  
        XtX_inv = np.linalg.inv(self.ols.X.T @ self.ols.X)
        #h = np.array([x @ np.linalg.inv(self.ols.X.T @ self.ols.X) @ x.T for x in X_new])
        h = np.einsum('ij,jk,ik->i', X_new, XtX_inv, X_new)  # ChatGPT's optimization for the above line of code which is otherwise too slow
        t_critical = t.ppf(1 - alpha / 2, self.ols.n - self.ols.p)
        se_pred = np.sqrt(self.ols.sigma_squared * (1 + h))
        lower_bounds = predictions - t_critical * se_pred
        upper_bounds = predictions + t_critical * se_pred
        return predictions, lower_bounds, upper_bounds
    
    def summary(self):
        if not self.ols.fitted:
            raise ValueError("Model is not yet fitted.")
        
        t_stats, p_values, significant = self.t_test()
    
        print("COEFFICIENT SUMMARY TABLE (Significance level 0.05)")
        print(f"{'Variable':<15}{'Estimated Coefficient':<25}{'T statistic':<15}{'P-value':<10}{'Significant':<10}")
        print(f"{'Intercept':<15}{self.ols.beta_hat[0]:<25.5f}{t_stats[0]:<15.5f}{p_values[0]:<10.5f}{str(significant[0]):<10}")
        for i in range(1, self.ols.p):
            print(f"{f'x{i}':<15}{self.ols.beta_hat[i]:<25.5f}{t_stats[i]:<15.5f}{p_values[i]:<10.5f}{str(significant[i]):<10}")


def main(): # usage code for testing / can add the examples to the docs
    
    #can fix the np.seed to have consistent results
    X = np.random.rand(100, 2)  
    y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) * 0.5  # linear model with gaussian noise

    # Fit the OLS
    ols = OLSModel(X, y)
    ols.fit()
    print("Beta coefficients:", ols.beta_hat)

    tester = Tester(ols)

    # Confidence intervals
    print("Confidence intervals:", tester.confidence_intervals())

    # t-tests
    t_stats, p_values, significant = tester.t_test()
    print("t-statistics:", t_stats)
    print("p-values:", p_values)
    print("Significant coefficients:", significant)

    # F-test
    R = np.eye(ols.p - 1, ols.p)  # Test all coefficients but intercept
    r = np.zeros(ols.p - 1)
    F_stat, F_p_value = tester.f_test(R, r)
    print("F-statistic:", F_stat)
    print("F-test p-value:", F_p_value)

    # Prediction intervals
    X_new = np.array([[0.5, 0.5], [0.2, 0.8]])
    predictions, lower_bounds, upper_bounds = tester.prediction_intervals(X_new)
    print("Predictions:", predictions)
    print("Prediction intervals:", np.column_stack((lower_bounds, upper_bounds)))

    #summary function
    tester.print_coefficient_summary()

if __name__ == "__main__":
    main()