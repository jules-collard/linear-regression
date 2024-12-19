import numpy as np
# import itertools
import regression
import random

class VariableSelector:
    def __init__(self, model_class, X, y, add_intercept=True, **model_kwargs):
        """
        Initialize the VariableSelector.

        :param model_class: A RegressionModel subclass (e.g., OLSModel, RidgeModel).
        :param X: Predictor variables (array).
        :param y: Target variable (array).
        :param add_intercept: Whether to add an intercept column to X.
        :param model_kwargs: Additional parameters for the model (e.g., lambda for RidgeModel).
        """
        self.model_class = model_class
        self.X = np.hstack((np.ones((X.shape[0], 1)), X)) if add_intercept else X
        self.y = y
        self.n, self.p = self.X.shape
        self.add_intercept = add_intercept
        self.model_kwargs = model_kwargs

        self.selected_covariates = []

    def fit_model(self, covariates, observations=None):
        """
        Fits the model in self.model_class for a specified subset of the data.
        :param covariates: list of indices.
        :param observations: list of indices (only used for cross-validation).
        :return: Fitted model.
        """
        if observations is None:
            observations = list(range(self.n))
        X_selected = self.X[observations, :]
        X_selected = X_selected[:, covariates]
        y_selected = self.y[observations]
        model = self.model_class(X_selected, y_selected, add_intercept=False, **self.model_kwargs)
        model.fit()
        return model
    
    def partition(self, K, list):
        random.shuffle(list)
        return [list[i::K] for i in range(K)]
    
    def compute_hold_out(self, covariates, ho_indices):
        complement = [i for i in list(range(self.n)) if i not in ho_indices ]
        n = len(complement)
        model = self.fit_model(covariates, ho_indices)
        y_test = self.y[complement]
        y_pred = model.predict(y_test)
        residuals = y_test - y_pred
        L_hold_out = sum(residuals ** 2) / n
        return L_hold_out
    
    def compute_cross_validation(self, covariates, K=10):
        partition = self.partition(K, list(range(self.n)))
        ho_sum = 0
        for l in partition:
            ho_sum += self.compute_hold_out(covariates, l)
        L_cv = ho_sum / K
        return L_cv
    
    def compute_criteria(self, covariates, k=10):
        model = self.fit_model(covariates)
        n, alpha = model.n, model.p
        sigma_hat = sum(model.residuals ** 2) / (n - 1)
        aic = n + (n * np.log(2 * np.pi * sigma_hat)) + (2 * alpha)
        bic = n + (n * np.log(2 * np.pi * sigma_hat)) + (np.log(n) * alpha)
        return {'AIC': aic, 'BIC': bic}

    def forward_selection(self, criterion='AIC', K=10, threshold=0):
        '''
        Perform forward variable selection using the specified criterion.

        :param criterion: Criterion to optimize ('AIC' for Akaike Information Criterion, 'BIC' for Bayesian Information Criterion, or 'CV' for Cross Validation).
        :param threshold: Minimum improvement required; used as stopping rule.
        :param K: Number of folds for cross validation.
        :return: Model fitted to the best covariate selection and list of selected covariate indices.
        '''
        best_covariates = []
        remaining_covariates = list(range(self.p))
        best_crit = float('inf')

        # Start with intercept if included
        if self.add_intercept:
            best_covariates.append(0) # change to add(0) if using set***
            remaining_covariates.remove(0)

        while remaining_covariates:
            results = [] # list of tuples containing (criterion value, covariate index)
            
            # Evaluate addition of each remaining covariate
            for covariate in remaining_covariates:
                candidate_model = best_covariates + [covariate]
                if criterion == 'CV':
                    crit_value = self.compute_cross_validation(candidate_model, K)
                else:
                    crit_value = self.compute_criteria(candidate_model)[criterion]
                results.append((crit_value, covariate))
            
            # Find the covariate that optimized the criterion
            results.sort()
            best_candidate_crit, best_candidate_covariate = results[0]

            if best_candidate_crit - best_crit < threshold:
                best_crit = best_candidate_crit
                best_covariates.append(best_candidate_covariate)
                remaining_covariates.remove(best_candidate_covariate)
            else:
                break  # Stop if improvement below threshold
        
        if criterion == 'AIC':
            self.best_forward_aic = best_crit
        elif criterion == 'BIC':
            self.best_forward_bic = best_crit
        
        self.selected_covariates = best_covariates
        best_model = self.fit_model(self.selected_covariates)

        return best_model, best_covariates
    
    def backward_selection(self, criterion='AIC', threshold=0):
        remaining_covariates = list(range(self.p))
        best_crit = float('inf')

        while remaining_covariates:
            results = [] # list of tuples containing (criterion value, covariate index)
            
            # Evaluate addition of each remaining covariate
            for covariate in remaining_covariates:
                candidate_model = [i for i in remaining_covariates if i != covariate ] # Temporarily remove covariate
                crit_value = self.compute_criteria(candidate_model)[criterion]
                results.append((crit_value, covariate))
            
            # Find the covariate that optimized the criterion
            results.sort()
            best_candidate_crit, best_candidate_covariate = results[0]

            if best_candidate_crit - best_crit < threshold:
                best_crit = best_candidate_crit
                remaining_covariates.remove(best_candidate_covariate)
            else:
                break  # Stop if improvement below threshold
        
        if criterion == 'AIC':
            self.best_backward_aic = best_crit
        elif criterion == 'BIC':
            self.best_backward_bic = best_crit
        
        self.selected_covariates = remaining_covariates
        best_model = self.fit_model(self.selected_covariates)

        return best_model, remaining_covariates
    
    
def test():
    X = np.random.rand(10, 5)  
    y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(10) * 0.5  # linear model with gaussian noise
    
    s = VariableSelector(regression.OLSModel, X, y)
    best_forward_model, best_forward_cov = s.forward_selection()
    best_backward_model, best_backward_cov = s.backward_selection()
    print("True model:", [0,1,2])
    print("Forward selection covariates under AIC:", best_forward_cov)
    print("Backward selection covariates under AIC:", best_backward_cov)

if __name__ == "__main__":
    test()