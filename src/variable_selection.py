import numpy as np
# import itertools
import regression
import random

class VariableSelector:
    def __init__(self, model_obj):
        """
        Initialize the VariableSelector.

        :param model_class: A RegressionModel subclass (e.g., OLSModel, RidgeModel).
        :param X: Predictor variables (array).
        :param y: Target variable (array).
        :param add_intercept: Whether to add an intercept column to X.
        :param model_kwargs: Additional parameters for the model (e.g., lambda for RidgeModel).
        """
        model_obj.check_fitted()
        self.model = model_obj
        self.model_class = model_obj.__class__
        # self.X = np.hstack((np.ones((X.shape[0], 1)), X)) if add_intercept else X
        # self.y = y
        # self.n, self.p = self.X.shape
        # self.add_intercept = add_intercept
        # self.model_kwargs = model_kwargs

        self.selected_covariates = []

    def fit_model(self, covariates, observations=None):
        """
        Fits the model in self.model_class for a specified subset of the data.
        :param covariates: list of indices.
        :param observations: list of indices (only used for cross-validation).
        :return: Fitted model.
        """
        if observations is None:
            observations = list(range(self.model.n))
        X_selected = self.model.X[observations, :]
        X_selected = X_selected[:, covariates]
        y_selected = self.model.y[observations]

        new_model = self.model_class(X_selected, y_selected, add_intercept=False)
        if self.model.hyperparameters is None:
            new_model.fit()
        else:
            new_model.fit(self.mode.hyperparameters)
        return new_model
    
    def partition(self, K, list):
        random.shuffle(list)
        return [list[i::K] for i in range(K)]
    
    def compute_hold_out(self, covariates, ho_indices):
        complement = [i for i in list(range(self.model.n)) if i not in ho_indices ]
        new_model = self.fit_model(covariates, complement)

        X_test = self.model.X[ho_indices, :]
        X_test = X_test[:, covariates]
        y_test = self.model.y[ho_indices]

        y_pred = new_model.predict(X_test, add_intercept=False)
        residuals = y_test - y_pred
        L_hold_out = sum(residuals ** 2) / len(ho_indices)
        return L_hold_out
    
    def compute_cross_validation(self, covariates, K=10):
        partition = self.partition(K, list(range(self.model.n)))
        ho_sum = 0
        for l in partition:
            ho_sum += self.compute_hold_out(covariates, l)
        L_cv = ho_sum / K
        return L_cv
    
    def compute_criteria(self, covariates, k=10):
        new_model = self.fit_model(covariates)
        n, alpha = new_model.n, new_model.p
        sigma_hat = sum(new_model.residuals ** 2) / (n - 1)
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
        remaining_covariates = list(range(self.model.p))
        best_crit = float('inf')

        # Start with intercept
        best_covariates.append(0)
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
        elif criterion == 'CV':
            self.best_forward_cv = best_crit
        
        self.selected_covariates = best_covariates
        best_model = self.fit_model(self.selected_covariates)

        return best_model
    
    def backward_selection(self, criterion='AIC', K=10, threshold=0):
        remaining_covariates = list(range(self.model.p))
        best_crit = float('inf')

        while remaining_covariates:
            results = [] # list of tuples containing (criterion value, covariate index)
            
            # Evaluate addition of each remaining covariate
            for covariate in remaining_covariates:
                candidate_model = [i for i in remaining_covariates if i != covariate ] # Temporarily remove covariate
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
                remaining_covariates.remove(best_candidate_covariate)
            else:
                break  # Stop if improvement below threshold
        
        if criterion == 'AIC':
            self.best_backward_aic = best_crit
        elif criterion == 'BIC':
            self.best_backward_bic = best_crit
        elif criterion == 'CV':
            self.best_forward_cv = best_crit
        
        self.selected_covariates = remaining_covariates
        best_model = self.fit_model(self.selected_covariates)

        return best_model
    
    
def test1():
    X = np.random.rand(100, 10) 
    beta = np.array([0, 2, 0, 6, 0, 0, 4, 8, 0, 0]) # true model: [0, 2, 4, 7, 8], where 0 is the intercept
    y = 3 + (X @ beta) + np.random.randn(100) * 0.5  # linear model with gaussian noise
    
    model = regression.OLSModel(X, y)
    model.fit()
    selector = VariableSelector(model)

    best_forward_model = selector.forward_selection(criterion='AIC')
    best_forward_covariates = selector.selected_covariates
    best_backward_model = selector.backward_selection(criterion='AIC')
    best_backward_covariates = selector.selected_covariates
    print("True model:", [0, 2, 4, 7, 8])
    print("Forward selection covariates under AIC:", best_forward_covariates)
    print("Backward selection covariates under AIC:", best_backward_covariates)

    best_forward_model = selector.forward_selection(criterion='BIC')
    best_forward_covariates = selector.selected_covariates
    best_backward_model = selector.backward_selection(criterion='BIC')
    best_backward_covariates = selector.selected_covariates
    print("True model:", [0, 2, 4, 7, 8])
    print("Forward selection covariates under BIC:", best_forward_covariates)
    print("Backward selection covariates under BIC:", best_backward_covariates)

    best_forward_model = selector.forward_selection(criterion='CV', K=100)
    best_forward_covariates = selector.selected_covariates
    best_backward_model = selector.backward_selection(criterion='CV', K=100)
    best_backward_covariates = selector.selected_covariates
    print("True model:", [0, 2, 4, 7, 8])
    print("Forward selection covariates under CV:", best_forward_covariates)
    print("Backward selection covariates under CV:", best_backward_covariates)

if __name__ == "__main__":
    test1()