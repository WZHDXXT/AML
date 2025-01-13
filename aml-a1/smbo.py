import ConfigSpace
import numpy as np
import pandas as pd
import typing

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel as C
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import norm
import scipy.stats as sps

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
        self.model = None
        self.R = []
        self.theta_inc = None
        self.theta_inc_performance = None

    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        self.R = capital_phi

        for config, performance in capital_phi:
            if self.theta_inc is None or performance < self.theta_inc_performance:
                self.theta_inc = config
                self.theta_inc_performance = performance

        self.fit_model()

    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        # Rs = []
        # for (c, p) in self.R:
        #     c = pd.DataFrame([c])
        #     p = [p]
        #     p = pd.DataFrame(p, columns=['score'])
        #     R_ = pd.concat([c, p], axis=1)
        #     Rs.append(R_)
        # df = pd.concat(Rs, axis=0)
        # X = df.iloc[:, :-1]
        # y = df.iloc[:, -1]
        df = pd.concat([pd.DataFrame([config]).assign(score=performance) for config, performance in self.R], axis=0)
        X = df.iloc[:, :-1]
        y = df['score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=['number']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
        ])
        # Create the ColumnTransformer to apply the appropriate transformations to each column
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # GaussianProcessRegressor as internal surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                                   normalize_y=True, random_state=40,
                                                   n_restarts_optimizer=30))
        ])
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"Internal Surrogate Model Evaluation: - MSE: {mse:.4f}, R2: {r2:.4f}")

        self.theta_inc = X.iloc[np.argmin(y)]
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
        config_space = list(self.config_space.values())
        random_configs = [dict(self.config_space.sample_configuration()) for _ in range(num_candidates)]
        for configs in random_configs:
            for config in config_space:
                if config.name not in configs.keys():
                    configs[config.name] = config.default_value
        
        configs_df = pd.DataFrame(random_configs)
        e = self.expected_improvement(self.model, self.theta_inc_performance, configs_df)
        best_index  = np.argmax(e)
        return random_configs[best_index]

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
        improvement = f_star - mu

        with np.errstate(divide='warn'):
            z = np.where(sigma > 0, improvement / sigma, 0)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0

        return ei.ravel()

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        config, performance = run
        self.R.append(run)

        if performance < self.theta_inc_performance:
            self.theta_inc = config
            self.theta_inc_performance = performance

        self.fit_model()
