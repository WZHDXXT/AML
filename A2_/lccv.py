import logging
import matplotlib.pyplot as plt
import numpy as np
import typing
from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        performance_diff = previous_performance - current_performance
        anchor_diff = previous_anchor - current_anchor

        project_factor = target_anchor - current_anchor
        optimistic_extrapolation = current_performance + project_factor*(performance_diff/anchor_diff)
        return optimistic_extrapolation

    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        evaluations = []
        extrapolated_values = []
        # first time
        if best_so_far == None:
            configuration['anchor_size'] = self.final_anchor
            performance = self.surrogate_model.predict(configuration)
            best_so_far = performance

        
        # first anchor_size
        current_anchor_size = self.minimal_anchor
        configuration['anchor_size'] = current_anchor_size
        performance = self.surrogate_model.predict(configuration)
        previous_performance = performance
        previous_anchor_size = current_anchor_size
        evaluations.append((previous_anchor_size, performance))
        current_anchor_size *= 2

        while current_anchor_size <= self.final_anchor:
            configuration['anchor_size'] = current_anchor_size
            performance = self.surrogate_model.predict(configuration)
            optimistic_extrapolation = self.optimistic_extrapolation(previous_anchor_size, previous_performance, current_anchor_size, performance, self.final_anchor)
            evaluations.append((current_anchor_size, performance))
            extrapolated_values.append(optimistic_extrapolation)
            
            if optimistic_extrapolation < best_so_far:
                # best_so_far = min(best_so_far, performance)
                previous_performance = performance
                previous_anchor_size = current_anchor_size
                current_anchor_size *= 2
            else:
                break
   
        return evaluations #, extrapolated_values

