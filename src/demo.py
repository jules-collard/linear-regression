import regression as reg
import aggregation
import numpy as np

#--DATA GENERATION--
X = np.random.rand(100, 2)  
y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) * 0.5  # linear model with gaussian noise

#############
# OLS Model #
#############
ols = reg.OLSModel(X, y)
ols.fit()

# Extract various aspects of the model
ols.coefficients()

tester = reg.OLS_Inference(ols)

# Bonferroni Confidence intervals
print("Confidence intervals:")
for interval in tester.confidence_intervals_bonferroni(0.025):
    print(interval)

# t-tests
t_stats, p_values, significant = tester.t_test()
print("t-statistics:", t_stats)
print("p-values:", p_values)
print("Significant coefficients:", significant)

# F-test
F_stat, F_p_value = tester.f_test_intercept_only()
print("F-statistic:", F_stat)
print("p:", F_p_value)

# Prediction intervals
X_new = np.array([[0.5, 0.5], [0.2, 0.8]])
predictions, intervals = tester.prediction_intervals(X_new)
print("Predictions:", predictions)
print("Prediction intervals:")
for interval in intervals:
    print(interval)

# View model summary
ols.summary()

#############
# WLS Model #
#############

wls = reg.WLSModel(X, y)
wls.fit()
wls.summary()

###############
# Ridge Model #
###############

ridge = reg.RidgeModel(X, y)
ridge.fit(5) # Must specify regularization paramter
ridge.summary()

###############
# Aggregation #
###############

agg = aggregation.Aggregator(ols, wls, ridge)
agg.fit()
agg.summary()
