import numpy as np
from scipy.stats import t, f
from abc import ABC, abstractmethod
from confints import ConfidenceInterval

class RegressionModel(ABC):
    """Abstract class representing a general regression model

    :param X: Design matrix (n x p)
    :type X: np.ndarray
    :param y: Response vector (n x 1)
    :type y: np.ndarray
    :param add_intercept: Option to add intercept column to X if not already included, defaults to True
    :type add_intercept: bool, optional
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, add_intercept=True):
        """Constructor method
        """
        self.fitted = False
        self.hyperparameters = None
        # Data
        self.X: np.ndarray = np.hstack((np.ones((X.shape[0], 1)), X)) if add_intercept else X
        self.y: np.ndarray = y
        self.n, self.p = self.X.shape

        # Model Output
        self.beta_hat: np.ndarray = None
        self.residuals: np.ndarray = None
        self.sigma_squared: np.ndarray = None
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

    def predict(self, X_new: np.ndarray, add_intercept=True) -> np.ndarray:
        """Generates prediction based on fitted model.

        :param X_new: Matrix data points for which predictions are made, with shape (n_new, p)
        :type X_new: np.ndarray
        :param add_intercept: Option to add intercept column to X if not already included, defaults to True
        :type add_intercept: bool, optional
        :raises ValueError: Model not fitted
        :return: Predicted values for the new data, shape (n_new,1)
        :rtype: np.ndarray
        """
        self.check_fitted()
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new)) if add_intercept else X_new
        return X_new @ self.beta_hat
    
    def summary(self):
        """Prints model summary.

        :raises ValueError: Model not fitted
        """
        self.check_fitted()
        print("COEFFICIENT SUMMARY TABLE")
        print(f"{'Intercept':<15}{self.beta_hat[0]:<25.5f}")
        for i in range( 1, self.p ): 
            print(f"{f'x{i -1}':<15}{self.beta_hat[i]:<25.5f}")
        print(f"R-squared: {self.compute_r2()}, Adjusted R-squared: {self.compute_adj_r2()}")

    def coefficients(self) -> np.ndarray:
        """Returns the vector of estimated coefficients for a linear model.

        :raises ValueError: Model not fitted
        :return: Vector of coefficients
        :rtype: np.ndarray
        """
        self.check_fitted()
        return self.beta_hat

    def compute_r2(self) -> float:
        """Calculates R^2 coefficient

        :raises ValueError: Model not fitted
        :return: R^2 coefficient
        :rtype: float
        """
        self.check_fitted()
        y_bar = np.mean(self.y)
        self.r2 = ((self.y_hat - y_bar).T @ (self.y_hat - y_bar)) / ((self.y - y_bar).T @ (self.y - y_bar))
        self.adj_r2 = 1 - ((1 - self.r2) * (self.n / (self.n - self.p)))
        return self.r2
    
    def compute_adj_r2(self) -> float:
        """Calculates adjusted R^2 coefficient

        :raises ValueError: Model not fitted
        :return: Adjusted R^2 coefficient
        :rtype: float
        """
        if self.adj_r2 is not None:
            self.compute_r2()
        return self.adj_r2

    def information_criteria(self) -> dict: # Implemented for OLS and Ridge only
        """Calculates the full-model AIC and BIC
        
        :raises ValueError: Model not fitted
        :return: Dictionary containing the AIC and BiC values
        :rtype: dict
        """
        self.check_fitted()
        self.aic = self.n + (self.n * np.log(2 * np.pi * self.sigma_squared)) + (2 * self.p)
        self.bic = self.n + (self.n * np.log(2 * np.pi * self.sigma_squared)) + (np.log(self.n) * self.p)
        return {'AIC': self.aic, 'BIC': self.bic}

    def check_fitted(self):
        """
        :meta private:
        """
        if not self.fitted: raise ValueError('Model not fitted')
        

#Implementation uses numpy for basic matrix operations.
#Also uses scipy to determine the quantiles of the t and f distribution
class OLSModel(RegressionModel):
    """Implementation of Ordinary Least Squares (OLS) regression. It provides methods to fit the model, predict new values, compute diagnostics, and calculate key metrics such as residuals, leverage values, and confidence intervals.

    :param X: Design matrix (n x p)
    :type X: np.ndarray
    :param y: Response vector (n x 1)
    :type y: np.ndarray
    :param add_intercept: Option to add intercept column to X if not already included, defaults to True
    :type add_intercept: bool, optional
    """
    
    def __init__(self, X, y, add_intercept=True):
        """Constructor method
        """
        RegressionModel.__init__(self, X, y, add_intercept=add_intercept)
        self.sigma_squared: float = None
        self.var_beta: np.ndarray = None # estimated variance of betahat

    # Calculate OLS estimator and resulting residuals
    def fit(self):
        """Fits OLS model to data (in place) by computing coefficients
        """
        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.beta_hat = XtX_inv @ self.X.T @ self.y
        self.fitted = True
        
        self.y_hat = self.predict(self.X, add_intercept=False)
        self.residuals = self.y - self.y_hat
        self.sigma_squared = (self.residuals.T @ self.residuals) / (self.n - self.p)
        self.var_beta = self.sigma_squared * XtX_inv

    def get_covariance_matrix(self) -> np.ndarray:
        self.check_fitted()
        return self.var_beta
    
    def hat_matrix(self) -> np.ndarray:
        self.check_fitted()
        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        H = self.X @ XtX_inv @ self.X.T
        return H
    
    def annihilator_matrix(self) -> np.ndarray:
        hat = self.hat_matrix()
        return np.eye(hat.shape[0], hat.shape[1]) - hat

    def leverages(self) -> np.ndarray:
        self.check_fitted()
        H = self.hat_matrix()
        return np.diag(H)

    def standardized_residuals(self) -> np.ndarray:
        """Computes standardized residuals for each observation.

        :raises ValueError: Model not fitted
        :return: Vector of standardized residuals, shape (n x 1)
        :rtype: np.ndarray
        """
        self.check_fitted()
        h = self.leverages()  # leverage values
        e = self.residuals  # residuals
        se = np.sqrt(self.sigma_squared * (1 - h))  # standard error
        standardized_residuals = e / se
        return standardized_residuals
    
    
    def summary(self):
        """Prints a summary of the OLS model, including the estimated coefficients and associated t-statistics, p-values, confidence intervals, and significance for each predictor.
        """
        self.check_fitted()
        infer = OLS_Inference(self)
        
        t_stats, p_values, significant = infer.t_test()
        confidences = infer.confidence_intervals_beta()
        F_stat, F_p_value = infer.f_test_intercept_only()
        self.compute_r2()
        self.information_criteria()

        print("\nCOEFFICIENT SUMMARY TABLE (Significance level 0.05)")
        print(f"{'Variable':<15}{'Estimated Coefficient':<25}{'T statistic':<15}{'P-value':<10}{'Significant':<13}{'Lower Bound ':<13}{'Upper Bound': <13}{'Coverage Level':<20}")
        print(f"{'Intercept':<15}{self.beta_hat[0]:<25.5f}{t_stats[0]:<15.5f}{p_values[0]:<10.5f}{str(significant[0]):<13}{confidences[0].lb:<15.5f}{confidences[0].ub:<13.5f}{confidences[0].coverage * 100:<15}")
        for i in range( 1, self.p ): 
            print(f"{f'x{i -1}':<15}{self.beta_hat[i]:<25.5f}{t_stats[i]:<15.5f}{p_values[i]:<10.5f}{str(significant[i]):<13}{confidences[i].lb:<15.5f}{confidences[i].ub:<13.5f}{confidences[i].coverage * 100:<15}")
        print("\nMODEL EVALUATION")
        print(f"F statistic: {F_stat}, p-value: {F_p_value}")
        print(f"R-squared: {self.r2}, Adjusted R-squared: {self.adj_r2}")
        print(f"Full-model AIC: {self.aic}")
        print(f"Full-model BIC: {self.bic}")

class OLS_Inference:
    """Class providing statistical inference methods for an OLS regression model, including confidence intervals for coefficients, t-tests, F-tests, and prediction intervals.
    
    :param ols: Instance of :class:OLSModel which has been fitted
    :type ols: OLSModel
    :raises ValueError: Model not fitted
    """
    def __init__(self, ols: OLSModel):
        """Constructor method
        """
        ols.check_fitted()
        self.ols = ols

    def confidence_intervals_beta(self, alpha=0.05) -> list[ConfidenceInterval]:
        """Computes the confidence intervals for each coefficient based on the t-distribution.

        :param alpha: Significance level for confidence intervals, defaults to 0.05
        :type alpha: float, optional
        :return: List of :class:ConfidenceInterval objects
        :rtype: list[ConfidenceInterval]
        """
        var_beta = self.ols.get_covariance_matrix()
        se_beta = np.sqrt(np.diag(var_beta))
        t_critical = t.ppf(1 - alpha / 2, self.ols.n - self.ols.p)
        lower_bounds = self.ols.beta_hat - t_critical * se_beta
        upper_bounds = self.ols.beta_hat + t_critical * se_beta
        #return np.column_stack((lower_bounds, upper_bounds))
        intervals = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            intervals.append(ConfidenceInterval(lb,ub, 1-alpha))
        
        return intervals

    def confidence_intervals_bonferroni(self, alpha=0.05) -> list[ConfidenceInterval]:
        """Applies the Bonferroni correction to the significance level by dividing significance level by the number of coefficients in the model. Computes the corrected confidence intervals for each coefficient.

        :param alpha: Significance level for confidence intervals, defaults to 0.05
        :type alpha: float, optional
        :return: List of :class:ConfidenceInterval objects
        :rtype: list[ConfidenceInterval]
        """
        # Bonferroni correction
        corrected_alpha = alpha / self.ols.p

        return self.confidence_intervals_beta(corrected_alpha)
    

    def t_statistics(self, hypothesized_values=None) -> np.ndarray: 
        """Computes t-statistic for each coefficient.

        :param hypothesized_values: The hypothesized values for each coefficient, defaults to vector of zeros.
        :type hypothesized_values: np.ndarray, optional
        :return: Vector of t-statistics
        :rtype: np.ndarray
        """
        if hypothesized_values is None:
            hypothesized_values = np.zeros_like(self.ols.beta_hat) # default: b_j = 0, for all j
        var_beta = self.ols.get_covariance_matrix()
        se_beta = np.sqrt(np.diag(var_beta))
        return (self.ols.beta_hat - hypothesized_values) / se_beta


    def t_test(self, hypothesized_values=None, alpha=0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a t-test for each coefficient to test if it is significantly different from the hypothesized value. Returns the t-statistics, p-values, and significance results.

        :param hypothesized_values:  The hypothesized values for each coefficient, defaults to vector of zeros.
        :type hypothesized_values: np.ndarray, optional
        :param alpha: Significance level for the tests, defaults to 0.05
        :type alpha: float, optional
        :return: Tuple with vectors of t-statistics, p-values and significance tests
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        t_stats = self.t_statistics(hypothesized_values=hypothesized_values)
       
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=self.ols.n - self.ols.p))
        # Evaluate significance
        significant = p_values < alpha
        return t_stats, p_values, significant

    #General F-test - can set the constraints as wanted
    def f_test(self, R: np.ndarray, r: np.ndarray) -> tuple[float, float]:
        """Performs F-test on general null hypothesis R @ beta_hat = r. Used to test multiple linear restrictions on the coefficients.

        :param R: Constraint matrix
        :type R: np.ndarray
        :param r: Right-hand side of constraints
        :type r: np.ndarray
        :return: Tuple with F-statistic and p-value
        :rtype: tuple[float, float]
        """
        # F-test for H0: Rβ = r
        R_beta_hat = R @ self.ols.beta_hat - r
        cov_R_beta = R @ self.ols.get_covariance_matrix() @ R.T
        F_stat = (R_beta_hat.T @ np.linalg.inv(cov_R_beta) @ R_beta_hat) / R.shape[0]
        F_stat /= self.ols.sigma_squared
        p_value = 1 - f.cdf(F_stat, R.shape[0], self.ols.n - self.ols.p)
        return F_stat, p_value
    
    #The more commonly used f-test to test if the model with only the intercept is correct
    def f_test_intercept_only(self) -> tuple[float, float]:
        """Performs an F-test for the null hypothesis that all coefficients except the intercept are zero.

        :return: Tuple with F-statistic and p-value
        :rtype: tuple[float, float]
        """
        R = np.eye(self.ols.p - 1, self.ols.p)  # Test all coefficients but intercept
        r = np.zeros(self.ols.p - 1)
        return self.f_test(R, r)

    def prediction_intervals(self, X_new :np.ndarray, add_intercept = True, alpha=0.05) -> tuple[np.ndarray, list[ConfidenceInterval]]:
        """Calculates prediction intervals for new observations

        :param X_new: Matrix of data points for which predictions are made, with shape (n_new, p)
        :type X_new: np.ndarray
        :param add_intercept: Option to add intercept column to X if not already included, defaults to True
        :type add_intercept: bool, optional
        :param alpha: Significance level for prediction intervals, defaults to 0.05
        :type alpha: float, optional
        :raises ValueError: Model not fitted
        :return: Tuple with vector of predictions and list of associated intervals
        :rtype: tuple[np.ndarray, list[ConfidenceInterval]]
        """
        predictions = self.ols.predict(X_new)
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new)) if add_intercept else X_new
        XtX_inv = np.linalg.inv(self.ols.X.T @ self.ols.X)
        #h = np.array([x @ np.linalg.inv(self.ols.X.T @ self.ols.X) @ x.T for x in X_new])
        h = np.einsum('ij,jk,ik->i', X_new, XtX_inv, X_new)  # ChatGPT's optimization for the above line of code which is otherwise too slow
        t_critical = t.ppf(1 - alpha / 2, self.ols.n - self.ols.p)
        se_pred = np.sqrt(self.ols.sigma_squared * (1 + h))
        lower_bounds = predictions - t_critical * se_pred
        upper_bounds = predictions + t_critical * se_pred
        #return predictions, lower_bounds, upper_bounds

        intervals = []
        for pred, lb, ub in zip(predictions, lower_bounds, upper_bounds):
            intervals.append(ConfidenceInterval(lb,ub, 1-alpha, estimate=pred))
        
        return predictions,intervals
    
    def confidence_intervals_mx(self, X_new: np.ndarray, add_intercept=True, alpha=0.05) -> tuple[np.ndarray, list[ConfidenceInterval]]:
        """Computes the confidence intervals for the regression function. The confidence intervals are computed considering only the variability in the regression function itself, not the residual error.

        :param X_new: Matrix of data points for which new regression function is calculated
        :type X_new: np.ndarray
        :param add_intercept: Option to add intercept column to X if not already included, defaults to True
        :type add_intercept: bool, optional
        :param alpha: Significance level for prediction intervals, defaults to 0.05
        :type alpha: float, optional
        :raises ValueError: Model not fitted
        :return: Tuple with vector of predictions and list of associated intervals
        :rtype: tuple[np.ndarray, list[ConfidenceInterval]]
        """
        self.ols.check_fitted()
        
        predictions = self.ols.predict(X_new)
        
        X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new)) if add_intercept else X_new
        
        XtX_inv = np.linalg.inv(self.ols.X.T @ self.ols.X)
        h = np.einsum('ij,jk,ik->i', X_new, XtX_inv, X_new) 
        t_critical = t.ppf(1 - alpha / 2, self.ols.n - self.ols.p)
        se_mx = np.sqrt(self.ols.sigma_squared * h)
        
        lower_bounds = predictions - t_critical * se_mx
        upper_bounds = predictions + t_critical * se_mx
        
        intervals = []
        for pred, lb, ub in zip(predictions, lower_bounds, upper_bounds):
            intervals.append(ConfidenceInterval(lb, ub, 1 - alpha, estimate=pred))
        
        return predictions, intervals
    

class WLSModel(RegressionModel):
    """Implementation of (feasible) WLS regression.
    
    :param X: Design matrix (n x p)
    :type X: np.ndarray
    :param y: Response vector (n x 1)
    :type y: np.ndarray
    :param add_intercept: Option to add intercept column to X if not already included, defaults to True
    :type add_intercept: bool, optional
    """

    def fit(self):
        """Fits WLS Model (in place) by estimating weights (known as feasible WLS).
        """
        # Fit OLS model
        ols = OLSModel(self.X, self.y, add_intercept=False)
        ols.fit()
        resid_squared = np.array([e ** 2 for e in ols.residuals])

        # Regress squared residuals against X to estimate weights
        resid_ols = OLSModel(self.X, resid_squared, add_intercept=False)
        resid_ols.fit()
        W = np.diag([1/e for e in resid_ols.y_hat])

        self.beta_hat = np.linalg.inv(self.X.T @ W @ self.X) @ self.X.T @ W @ self.y
        self.fitted = True

        self.y_hat = self.predict(self.X, add_intercept=False)
        self.residuals = self.y - self.y_hat
        

class RidgeModel(RegressionModel):
    """Implementation of Ridge estimator, requiring selection of regularization parameter.
    
    :param X: Design matrix (n x p)
    :type X: np.ndarray
    :param y: Response vector (n x 1)
    :type y: np.ndarray
    :param add_intercept: Option to add intercept column to X if not already included, defaults to True
    :type add_intercept: bool, optional
    """

    def fit(self, ridge_lambda: float):
        """Computes ridge estimator (in place)..

        :param ridge_lambda: Regularization parameter (must be greater)
        :type ridge_lambda: float
        :raises ValueError: Regularization parameter must be greater than 0
        """
        if ridge_lambda <= 0:
            raise ValueError("Regularization parameter must be greater than 0")
        self.beta_hat = np.linalg.inv((self.X.T @ self.X) + (ridge_lambda * np.identity(self.p))) @ self.X.T @ self.y
        self.fitted = True
        self.hyperparameters = ridge_lambda

        self.y_hat = self.predict(self.X, add_intercept=False)
        self.residuals = self.y - self.y_hat
        self.sigma_squared = sum(self.residuals ** 2 ) / (self.n - np.trace(self.hat_matrix()))

    def hat_matrix(self):
        self.check_fitted()
        H = self.X @ (np.linalg.inv((self.X.T @ self.X) + (self.hyperparameters * np.identity(self.p)))) @ self.X.T
        return H