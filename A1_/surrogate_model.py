
import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # fit the external surrogate model using OneHotEncoder
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        numerical_features = x.select_dtypes(include=['number']).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', sklearn.impute.SimpleImputer(strategy='mean'), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=1000, random_state=40))
        ])
        model.fit(x_train, y_train)
        self.model = model

        pred = model.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        corr, _ = spearmanr(y_test, pred)
        print(f"External surrogate Model Evaluation: - MSE: {mse:.4f}, R2: {r2:.4f}")
        print(f"Spearman Correlation: {corr:.4f}")


    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        if self.model == None:
            raise ValueError("Train the model first.")
        # fill the keys without full dictionary with default values of the config space
        else:
            for key in dict(self.config_space).keys():
                if key not in theta_new.keys():
                    theta_new[key] = dict(self.config_space)[key].default_value
            list_theta = [theta_new]
            x_test = pd.DataFrame(list_theta)
            x_test = x_test[self.df.keys()[:-1]]
        # predict the performance as ground truth

            y_test = self.model.predict(x_test)
            return y_test[0]