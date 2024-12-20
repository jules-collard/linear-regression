import numpy as np
import regression

class VariableSelector:
    def __init__(self, model_obj):
        """Initialize the VariableSelector.
        """
        model_obj.check_fitted() # The only reason that the model must be fitted is to get ridge_lambda. Maybe remove?
        self.model = model_obj
        self.model_class = model_obj.__class__

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
        new_model = self.model_class(X_selected, y_selected, add_intercept=False) # Fit model on subset data
        # Check if model requires hyperparameter (i.e., is Ridge)
        if self.model.hyperparameters is None:
            new_model.fit()
        else:
            new_model.fit(self.mode.hyperparameters)
        return new_model
    
    def partition(self, K, list):
        list = np.random.permutation(list).tolist()
        return [list[i::K] for i in range(K)]
    
    def compute_hold_out(self, covariates, ho_indices):
        # Training
        complement = [i for i in list(range(self.model.n)) if i not in ho_indices ]
        new_model = self.fit_model(covariates, complement)
        # Testing
        X_test = self.model.X[ho_indices, :]
        X_test = X_test[:, covariates]
        y_test = self.model.y[ho_indices]
        y_hat = new_model.predict(X_test, add_intercept=False)
        residuals = y_test - y_hat
        L_hold_out = sum(residuals ** 2) / len(ho_indices)
        return L_hold_out
    
    def compute_cross_validation(self, covariates, K=10):
        partition = self.partition(K, list(range(self.model.n)))
        ho_sum = 0
        for l in partition:
            ho_sum += self.compute_hold_out(covariates, l)
        L_cv = ho_sum / K
        return L_cv
    
    def compute_ic(self, covariates):       
        new_model = self.fit_model(covariates)
        return new_model.information_criteria()

    def forward_selection(self, criterion='AIC', K=10, threshold=0):
        """
        Perform forward variable selection using the specified criterion.

        :param criterion: Criterion to optimize ('AIC' for Akaike Information Criterion, 'BIC' for Bayesian Information Criterion, or 'CV' for Cross Validation).
        :param threshold: Minimum improvement required; used as stopping rule.
        :param K: Number of folds for cross validation.
        :return: Model fitted to the best covariate selection and list of selected covariate indices.
        """
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
                    crit_value = self.compute_ic(candidate_model)[criterion] # Extract value with dictionary key
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
        """
        Perform backward variable selection using the specified criterion.

        :param criterion: Criterion to optimize ('AIC' for Akaike Information Criterion, 'BIC' for Bayesian Information Criterion, or 'CV' for Cross Validation).
        :param threshold: Minimum improvement required; used as stopping rule.
        :param K: Number of folds for cross validation.
        :return: Model fitted to the best covariate selection and list of selected covariate indices.
        """
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
                    crit_value = self.compute_ic(candidate_model)[criterion] # Extract value with dictionary key
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