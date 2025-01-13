import logging
import numpy as np
import pandas as pd
import typing
from scipy.optimize import curve_fit
from vertical_model_evaluator import VerticalModelEvaluator


class IPL(VerticalModelEvaluator):

    @staticmethod
    def probability_extrapolation(x, c, a, alpha) -> float:
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

        '''print(f"Anchor sizes (x): {x}")
        print(f"Parameters: c={c}, a={a}, alpha={alpha}")'''
        # pow3
        # op_perf = c - a * x ** (-alpha)
        op_perf = c + a * x ** (-alpha)
        # vapor pressure
        # op_perf = np.exp(kwargs["a"] + kwargs["b"] / kwargs["x"] + kwargs["c"] * np.log(kwargs["x"]))
        return op_perf

    # def fit(self, anchor_sizes, performances):
    #     init_param = [np.min(performances), 0.1, 0.5]
    #     # init_param = [np.mean(performances), 0.01, 0.05]
    #     opt_param, cov_mat_param = curve_fit(
    #         self.probability_extrapolation,
    #         anchor_sizes,
    #         performances,
    #         p0=init_param,
    #         maxfev=10000,
    #         # bounds=([0, 0, 0], [1, 1, 10]),
    #     )
    #     print(opt_param)
    #     return opt_param

    def fit(self, anchor_sizes, performances):
        anchor_sizes = anchor_sizes
        performances = performances
        best_popt = None
        best_error = float("inf")

        initial_c = np.mean(performances)
        # initial_c = np.min(performances)
        initial_guesses = [
            [initial_c, 1.0, 0.7],
            [initial_c * 0.8, 0.5, 0.5],
            [initial_c * 1.2, 1.0, 0.5],
            # [performances[-1], performances[1] - performances[0], 0.5],
            # [performances[-1], 1.0, 0.5],
            # [performances[-1] * 0.8, 2.0, 0.7],
            # [performances[-1] * 1.2, 0.5, 0.3],
        ]
        # choose the best initial_guesses
        for initial_guess in initial_guesses:
            # try:
            popt, _ = curve_fit(
                self.probability_extrapolation,
                anchor_sizes,
                performances,
                p0=initial_guess,
                maxfev=50000,
                # bounds=([0, 0, 0], [1, 1, 10]),
            )

            residuals = performances - self.probability_extrapolation(
                anchor_sizes, *popt
            )
            error = np.sum(residuals**2)

            if error < best_error:
                best_error = error
                best_popt = popt
            # except RuntimeError:
            #     continue
        if best_popt is None:
            best_popt = initial_guesses[0]
        return best_popt

    # self.c, self.a, self.alpha = best_popt

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

        percentage_datasets = 0.5
        anchor_sizes = [
            self.surrogate_model.x.anchor_size[x]
            for x in range(
                int(
                    len(self.surrogate_model.x.anchor_size.unique())
                    * percentage_datasets
                )
            )
        ]
        performances = []
        results = []
        for i in range(len(anchor_sizes)):

            configuration["anchor_size"] = anchor_sizes[i]
            performances.append(self.surrogate_model.predict(configuration))

            results.append((anchor_sizes[i], performances[i]))

        c, a, alpha = self.fit(anchor_sizes=anchor_sizes, performances=performances)
        x = self.final_anchor
        estimated_performance = self.probability_extrapolation(
            c=c, a=a, x=x, alpha=alpha
        )

        if estimated_performance > best_so_far:

            return results
        else:
            configuration["anchor_size"] = self.final_anchor
            final_performance = self.surrogate_model.predict(configuration)
            results.append((self.final_anchor, final_performance))
            return results
            """
            i = len(anchor_sizes)

            anchor_sizes.extend(
                [
                    x
                    for x in self.surrogate_model.x.anchor_size.unique()
                    if x > 256
                ]
            )

            while i < len(anchor_sizes):
                current_anchor = anchor_sizes[i]
                configuration["anchor_size"] = current_anchor
                current_performance = self.surrogate_model.predict(configuration)
                performances.append(current_performance)
                c, a, alpha = self.fit(
                    anchor_sizes=anchor_sizes[: i + 1],
                    performances=performances[: i + 1],
                )
                estimated_performance = estimated_performance = (
                    self.probability_extrapolation(c=c, a=a, x=x, alpha=alpha)
                )

                results.append((current_anchor, current_performance))

                if best_so_far is not None and estimated_performance >= best_so_far:
                    break

                i += 1

            return results
            """
