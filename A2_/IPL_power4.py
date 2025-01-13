import numpy as np
import typing
from vertical_model_evaluator import VerticalModelEvaluator
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class IPL(VerticalModelEvaluator):
    def __init__(self, surrogate_model: Pipeline, minimal_anchor: int, final_anchor: int):
        super().__init__(surrogate_model, minimal_anchor, final_anchor)
        self.c = None
        self.a = None
        self.b = None
        self.alpha = None
    
    @staticmethod
    def pow4_model(x, c, a, b, alpha):
        # Ensure x is a numpy array for safe element-wise operations
        x = np.array(x)
        return c - (-a * x + b) ** (-alpha)
    
    def fit(self, evaluations: typing.List[typing.Tuple[int, float]]):
        anchor_sizes, performances = zip(*evaluations)
        best_popt = None
        best_error = float('inf')

        initial_c = np.mean(performances)
        initial_guesses = [
            [0.769, 1.0, 1.0, 0.7],  # Baseline guess
            [0.692, 0.5, 0.5, 0.5],  # Lower guess for c and smoother curvature
            [0.846, 1.5, 1.2, 1.0],  # Higher guess for c with steep slope
            [0.769, 0.8, 1.5, 0.6],  # Adjusting offset and curvature
            [0.769, 1.2, 0.8, 0.4]   # Different slope and decay
        ]
        for initial_guess in initial_guesses:
            try:
                popt, _ = curve_fit(self.pow4_model, anchor_sizes, performances, p0=initial_guess)
                residuals = performances - self.pow4_model(anchor_sizes, *popt)
                error = np.sum(residuals ** 2)
                if error < best_error:
                    best_error = error
                    best_popt = popt
            except RuntimeError:
                continue

        if best_popt is None:
            best_popt = initial_guesses[0] 
            print("Warning: No optimal parameters found. Using default initial guess.")

        self.c, self.a, self.b, self.alpha = best_popt

    def predict(self, target_anchor: int) -> float:
        if self.c is None or self.a is None or self.b is None or self.alpha is None:
            raise ValueError("The model must be fitted before making predictions.")
        predicted_performance = self.pow4_model(target_anchor, self.c, self.a, self.b, self.alpha)
        return predicted_performance
    '''def plot_fitted_curve(self, anchor_sizes, performances):
        x_fit = np.linspace(min(anchor_sizes), max(anchor_sizes), 100)
        y_fit = self.pow4_model(x_fit, self.c, self.a, self.b, self.alpha)

        plt.scatter(anchor_sizes, performances, color='blue', label='Actual Data')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
    '''
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:

        anchor_schedule = [64, 128, 256, 512, 1024]
        score = []
        # Determine a fixed learning curve schedule for every configuration 
        # (e.g., [16, 32, 64, 128, 256]) and fit an IPL to the learning curve data.
        for current_anchor in anchor_schedule:
            configuration['anchor_size'] = current_anchor
            performance = self.surrogate_model.predict(configuration) # Ensure single value
            score.append((current_anchor, performance))
        
        # print(score)
        self.fit(score)

        # Extrapolate the performance for final_anchor
        ipl_extrapolation = self.predict(self.final_anchor)
        # print(ipl_extrapolation)

        # Update the model performance based on extrapolation
        configuration['anchor_size'] = self.final_anchor
        performance = self.surrogate_model.predict(configuration)  # Ensure single value
        
        if best_so_far is None:
            best_so_far = performance
            score.append((self.final_anchor, performance))
        elif ipl_extrapolation < best_so_far:
            best_so_far = performance
            score.append((self.final_anchor, performance))
        
        return score