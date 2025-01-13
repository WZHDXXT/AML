import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from smbo import SequentialModelBasedOptimization
from surrogate_model import SurrogateModel
from random_search import RandomSearch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    parser.add_argument('--configurations_performance_file_1', type=str, default='config_performances_dataset-6.csv')
    parser.add_argument('--configurations_performance_file_2', type=str, default='config_performances_dataset-11.csv')
    parser.add_argument('--configurations_performance_file_3', type=str, default='config_performances_dataset-1457.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=50)

    return parser.parse_known_args()[0]

def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df_list = []
    df_1 = pd.read_csv(args.configurations_performance_file)
    df_2 = pd.read_csv(args.configurations_performance_file_1)
    df_3 = pd.read_csv(args.configurations_performance_file_2)
    df_4 = pd.read_csv(args.configurations_performance_file_3)
    df_list.append(df_1)
    df_list.append(df_2)
    df_list.append(df_3)
    df_list.append(df_4)
    # df = pd.read_csv(args.configurations_performance_file)
    for df in df_list:
        # successive halving
        halving_factor = 2
        surrogate_model_halving = SurrogateModel(config_space)
        surrogate_model_halving.fit(df)

        results = {
            'successive halving': [1.0]
        }
        best_performance = []
        best = np.inf
        for idx in range(args.num_iterations):
            total_anchor = [0]
            total_performance = {}

            # anchor size to be 10% of data
            current_anchor = 25
            configs = [dict(config_space.sample_configuration()) for _ in range(64)]
            while(current_anchor<=args.max_anchor_size):
                total_anchor.append(current_anchor)
                performances = []
                m = len(configs)

                for config in configs:
                    for hp in list(config_space.values()):
                        if hp.name not in config.keys():
                            config[hp.name] = hp.default_value
                    config['anchor_size'] = current_anchor

                    performance = surrogate_model_halving.predict(config)
                    performances.append(performance)

                total_performance['M'+str(current_anchor)] = performances
                performances = np.array(performances)
                performances_sorted = np.argsort(performances)
                num = int(m/halving_factor)

                configs = [configs[index] for index in performances_sorted[:num]]
                current_anchor *= halving_factor
            p = total_performance.get(list(total_performance.keys())[-1])[0]
            if p < best:
                best = p
                best_performance.append(total_performance)

            results['successive halving'].append(min(results['successive halving'][-1], best))

        # random search
        random_search = RandomSearch(config_space)
        surrogate_model_random = SurrogateModel(config_space)
        surrogate_model_random.fit(df)

        results_randomsearch = {
            'random_search': [1.0]
        }

        for idx in range(args.num_iterations):
            theta_new = dict(random_search.select_configuration())
            theta_new['anchor_size'] = args.max_anchor_size
            performance = surrogate_model_random.predict(theta_new)
            # ensure to only record improvements
            results_randomsearch['random_search'].append(min(results_randomsearch['random_search'][-1], performance))
            random_search.update_runs((theta_new, performance))
        print('random search', min(results_randomsearch['random_search']))

        # smbo
        surrogate_model_smbo = SurrogateModel(config_space)
        surrogate_model_smbo.fit(df)

        smbo = SequentialModelBasedOptimization(config_space)
        initial_runs = []

        configs = [dict(config_space.sample_configuration()) for _ in range(200)]
        for config in configs:
            config['anchor_size'] = args.max_anchor_size
            config['score'] = surrogate_model_smbo.predict(config)

        configs_ = configs.copy()
        for c in configs_:
            c = c.pop('anchor_size')
        variables = list(df.keys())
        variables.remove('anchor_size')
        df_config = pd.DataFrame([[i[j] for j in variables] for i in configs_], columns=variables)
        m, n = df_config.shape

        for _, row in df_config.iterrows():
            config = dict(row.head(n-1))
            p = row['score']
            initial_runs.append((config, p))
        smbo.initialize(initial_runs)

        results_BO = {
            'BO': [1.0]
            }

        for idx in range(args.num_iterations):
            smbo.fit_model()
            theta_new = smbo.select_configuration()
            theta_new['anchor_size'] = args.max_anchor_size
            theta_new_ = theta_new.copy()
            performance = surrogate_model_smbo.predict(theta_new)
            del theta_new_['anchor_size']
            results_BO['BO'].append(min(results_BO['BO'][-1], performance))
            # results_BO['BO'].append(min(results_BO['BO'][-1], performance))
            smbo.update_runs((theta_new_, performance))

        print("results BO", min(results_BO['BO']))

        plt.plot(range(len(results_randomsearch['random_search'])), results_randomsearch['random_search'], color='blue', label='random search')
        plt.plot(range(len(results['successive halving'])), results['successive halving'], color='red', label='successive halving')
        plt.plot(range(len(results_BO['BO'])), results_BO['BO'], color='green', label='smbo')
        plt.legend()

        plt.yscale('log')
        plt.show()


if __name__ == '__main__':
    run(parse_args())