import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel


import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
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
    colors = ['red', 'magenta', 'green', 'orange']
    i = 0

    for df in df_list:
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
      plt.plot(range(len(results_randomsearch['random_search'])), results_randomsearch['random_search'], color=colors[i], label='dataset'+str(i))
      i += 1
    plt.yscale('log')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    run(parse_args())