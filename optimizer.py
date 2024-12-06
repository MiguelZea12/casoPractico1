from scipy.optimize import minimize
import numpy as np

def optimize_budget(model, poly, budget=300):
    def sales_prediction(budget_allocation):
        budget = np.array(budget_allocation).reshape(1, -1)
        budget_poly = poly.transform(budget)
        return -model.predict(budget_poly)[0]  # Maximizar ventas

    bounds = [(0, budget)] * 3
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - budget},
        {"type": "ineq", "fun": lambda x: x[0] - 50},
        {"type": "ineq", "fun": lambda x: x[2] - 30},
    ]
    initial_guess = [budget / 3] * 3

    result = minimize(sales_prediction, initial_guess, bounds=bounds, constraints=constraints)
    return result.x, -result.fun
