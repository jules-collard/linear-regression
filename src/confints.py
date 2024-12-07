# Confidence intervals for coefficients, predictions
class ConfidenceInterval:

    # Simple initialisation
    def __init__(self, estimate: float, lower_bound: float, upper_bound: float, coverage: float):
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
    
    def get_bounds(self) -> list[float]:
        return [self.lb, self.ub]