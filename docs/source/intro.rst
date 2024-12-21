Introduction
============

``linear_regression_ols`` is a Python package implementing the most commonly used linear regression techniques, including Ordinary Least Squares, Weighted Least Squares, Ridge Estimation and Cross-Validation. In the OLS case, functionality is provided for inference and model diagnostics, including hypothesis testing, confidence intervals and the AIC and BIC. Furthermore, forwards and backwards variable selection with respect to both the AIC and BIC are implemented. Implementation uses `numpy` and `scipy` for linear algebra and distributions.

Full source code, README and wiki is available at the repository https://github.com/jules-collard/linear-regression-ols.

Usage
=====
General usage philosophy is as follows:

1. Select a model from `OLSModel`, `WLSModel`, `RidgeModel` or `Aggregator`. These all inherit from parent class `RegressionModel`.
2. Initialise model object with data, for example `ols = OLSModel(X,y)`
3. Fit model with `ols.fit()`

From this point, certain methods are common to all models, inheriting from the ``RegressionModel`` class:

* ``summary()`` for a model summary
* ``predict`` for predictions on new data
* ``compute_r2()``, ``compute_adj_r2()``, ``information_criteria()`` for model diagnostics

See documentation on ``variable_selection`` submodule for variable selection techniques, and the ``OLS_Inference`` class for inference techniques.

See the examples section of this documentation, or ``demo.py`` in the source code, for full implementation examples.