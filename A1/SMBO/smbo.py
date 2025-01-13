import ConfigSpace
import pandas as pd
import numpy as np
import typing

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as sps
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel as C

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class SequentialModelBasedOptimization(object):

    def __init__(self, config_space):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space
        self.R = []
        self.theta_inc = None
        self.theta_inc_performance = None
        self.model = None

    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        for (c, p) in capital_phi:
            self.R.append((c, p))

            if self.theta_inc is None or p < self.theta_inc_performance:
                self.theta_inc = c
                self.theta_inc_performance = p
    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        # key = list(dict(self.config_space).keys())
        Rs = []
        for (c, p) in self.R:
            c = pd.DataFrame([c])
            p = [p]
            p = pd.DataFrame(p, columns=['score'])
            R_ = pd.concat([c, p], axis=1)
            Rs.append(R_)
        df = pd.concat(Rs, axis=0)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        train_in, test_in, train_out, test_out = train_test_split(x, y, test_size=0.2)
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        numerical_features = x.select_dtypes(include=['number']).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[('num',
                           Pipeline([
                               ('imputer', SimpleImputer(strategy='mean')),
                               ('scaler', StandardScaler())
                               ]), numerical_features),
                               ('cat', OneHotEncoder(), categorical_features)
                               ])

        # GaussianProcessRegressor as internal surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                                   normalize_y=True, random_state=40,
                                                   n_restarts_optimizer=30))])
        model.fit(train_in, train_out)
        self.model = model

        pred = self.model.predict(test_in)
        mse = mean_squared_error(test_out, pred)
        r2 = r2_score(test_out, pred)
        print(f"Internal Surrogate Model Evaluation: - MSE: {mse:.4f}, R2: {r2:.4f}")

        self.theta_inc = x.iloc[np.argmin(y)]
        self.theta_inc_performance = y.iloc[np.argmin(y)]

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """

        num_candidates = 200
        # smbo.select configuration(sample configurations(many))
        configs = [dict(self.config_space.sample_configuration()) for _ in range(num_candidates)]
        for config in configs:
            for hp in list(self.config_space.values()):
                if hp.name not in config.keys():
                    config[hp.name] = hp.default_value

        configs_df = pd.DataFrame(configs)
        e = self.expected_improvement(self.model, self.theta_inc_performance, configs_df)
        configure = configs[np.argmax(e)]
        return configure

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        improvement = f_star - mu

        with np.errstate(divide='warn'):
            Z = np.where(sigma > 0, improvement / sigma, 0)
            ei = improvement * sps.norm.cdf(Z) + sigma * sps.norm.pdf(Z)
            ei[sigma == 0.0] == 0.0
        return ei.ravel()

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        (c, p) = run

        self.R.append((c, p))

        if p < self.theta_inc_performance:
            self.theta_inc = c
            self.theta_inc_performance = p