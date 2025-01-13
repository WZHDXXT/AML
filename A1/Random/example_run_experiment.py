import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    random_search = RandomSearch(config_space)

    surrogate_model_random = SurrogateModel(config_space)
      # train surrogate model on given configuration space
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

    print(min(results_randomsearch['random_search']))

if __name__ == '__main__':
    run(parse_args())
