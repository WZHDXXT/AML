import numpy as np
import matplotlib.pyplot as plt
import typing
from vertical_model_evaluator import VerticalModelEvaluator
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit

class IPL(VerticalModelEvaluator):
    def __init__(self, surrogate_model: Pipeline, minimal_anchor: int, final_anchor: int):
        super().__init__(surrogate_model, minimal_anchor, final_anchor)
        self.c = None
        self.a = None
        self.alpha = None
    
    @staticmethod
    def pow3_model(x, c, a, alpha):
        return c - a * x ** (-alpha)
    
    def fit(self, evaluations: typing.List[typing.Tuple[int, float]]):
        anchor_sizes, performances = zip(*evaluations)
        best_popt = None
        best_error = float('inf')
        initial_c = np.min(performances)
        initial_guesses = [
            [initial_c, 1.0, 0.7],
            [initial_c * 0.8, 0.5, 0.5],
            [initial_c * 1.2, 1.0, 0.5]
        ]
        # choose the best initial_guesses
        for initial_guess in initial_guesses:
            try:
                popt, _ = curve_fit(self.pow3_model, anchor_sizes, performances, p0=initial_guess, maxfev=10000)
                
                residuals = performances - self.pow3_model(anchor_sizes, *popt)
                error = np.sum(residuals ** 2)

                if error < best_error:
                    best_error = error
                    best_popt = popt
            except RuntimeError:
                continue
        if best_popt is None:
            best_popt = initial_guesses[0]
        self.c, self.a, self.alpha = best_popt

    def predict(self, target_anchor: int) -> float:
        if self.c is None or self.a is None or self.alpha is None:
            raise ValueError("The model must be fitted before making predictions.")
        predicted_performance = self.pow3_model(target_anchor, self.c, self.a, self.alpha)
        return predicted_performance
    
    def plot_fitted_curve(self, anchor_sizes, performances):
        x_fit = np.linspace(min(anchor_sizes), max(anchor_sizes), 100)
        y_fit = self.pow3_model(x_fit, self.c, self.a, self.alpha)

        plt.scatter(anchor_sizes, performances, color='blue', label='Actual Data')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.legend()
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:

        anchor_schedule = [16, 32, 64, 128, 256, 512, 1024]
        score = []
        # Determine a fixed learning curve schedule for every configuration 
        # (e.g., [16, 32, 64, 128, 256]) and fit an IPL to the learning curve data.
        for i in range(len(anchor_schedule)):
            current_anchor = anchor_schedule[i]
            configuration['anchor_size'] = current_anchor
            performance = self.surrogate_model.predict(configuration)            
            score.append((current_anchor, performance))
        # print(score)
        self.fit(score)


        ipl_extrapolation = self.predict(self.final_anchor)
        # print(ipl_extrapolation)
        # evaluations.append((current_anchor_size, performance))
        configuration['anchor_size'] = self.final_anchor
        performance = self.surrogate_model.predict(configuration)
        if best_so_far == None or ipl_extrapolation < best_so_far:
            # best_so_far = performance
            score.append((self.final_anchor, performance))
        
        return score