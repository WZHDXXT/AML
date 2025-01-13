import logging
import numpy as np
import pandas as pd
import typing
import math
from vertical_model_evaluator import VerticalModelEvaluator


class LCCV(VerticalModelEvaluator):

    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int,
        previous_performance: float,
        current_anchor: int,
        current_performance: float,
        target_anchor: int,
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
        op_perf = current_performance + (target_anchor - previous_anchor) * (
            previous_performance - current_performance
        ) / (previous_anchor - current_anchor)

        return op_perf

    def evaluate_model(
        self, best_so_far: typing.Optional[float], configuration: typing.Dict
    ) -> typing.List[typing.Tuple[int, float]]:
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
        if best_so_far is None:
            configuration["anchor_size"] = self.final_anchor
            best_so_far = self.surrogate_model.predict(configuration)

        anchor_sizes = [
            x
            for x in self.surrogate_model.x.anchor_size.unique()
            if x >= self.minimal_anchor
        ]
        
        '''anchor_sizes = [
            25 * (2 ** i) 
            for i in range(int(math.log(self.final_anchor / 25, 2)) + 1)
        ]'''

        # anchor_sizes = [x for x in range(self.minimal_anchor, self.final_anchor)]
        configuration["anchor_size"] = anchor_sizes[0]
        performances = [self.surrogate_model.predict(configuration)]
        results = [(anchor_sizes[0], performances[0])]

        i = 1
        while i < len(anchor_sizes):
            current_anchor = anchor_sizes[i]
            configuration["anchor_size"] = current_anchor
            current_performance = self.surrogate_model.predict(configuration)
            performances.append(current_performance)

            previous_anchor = anchor_sizes[i - 1]
            previous_performance = performances[i - 1]

            op_perf = self.optimistic_extrapolation(
                previous_anchor=previous_anchor,
                previous_performance=previous_performance,
                current_anchor=current_anchor,
                current_performance=current_performance,
                target_anchor=self.final_anchor,
            )

            results.append((current_anchor, current_performance))

            if best_so_far is not None and op_perf >= best_so_far:
                break

            i += 1

        return results
    
    
