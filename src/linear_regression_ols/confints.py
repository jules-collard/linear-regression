# Confidence intervals for coefficients, predictions
class ConfidenceInterval:
    """Class representing confidence intervals. Stores the estimate, lower and upper bounds, and coverage of the interval. Call print() on the object to view the intervals, or use the implemented getters and setters.

    :param lower_bound: Lower bound of interval
    :type lower_bound: float
    :param upper_bound: Upper bound of interval
    :type upper_bound: float
    :param coverage: Coverage of interval (between 0 and 1)
    :type coverage: float
    :param estimate: Estimated parameter, defaults to None
    :type estimate: _type_, optional
    :raises ValueError: Bounds must be well-defined, with estimate within bounds
    """

    # Simple initialisation
    def __init__(self, lower_bound: float, upper_bound: float, coverage: float, estimate=None):
        """Constructor method
        """
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be strictly less than upper bound")
        elif estimate is not None and (estimate <= lower_bound or estimate >= upper_bound):
            raise ValueError("Estimate must be within bounds")
        self.estimate: float = estimate
        self.lb: float = lower_bound
        self.ub: float = upper_bound
        self.coverage: float = coverage

    
    def __str__(self):
        return f"{self.coverage*100}% Confidence Interval: \n [{self.lb}, {self.ub}]"
    
    # Getters & setters
    def get_ub(self) -> float:
        return self.ub
    
    def get_lb(self) -> float:
        return self.lb
    
    def get_estimate(self) -> float:
        return self.estimate
    
    def get_bounds(self) -> list[float]:
        return [self.lb, self.ub]