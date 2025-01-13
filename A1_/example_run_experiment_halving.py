import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_model import SurrogateModel
import numpy as np

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

    return parser.parse_args()


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

    resultss = []
    for df in df_list:
      halving_factor = 2
      surrogate_model = SurrogateModel(config_space)
      surrogate_model.fit(df)

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

                  performance = surrogate_model.predict(config)
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

      resultss.append(results)
      colors = ['red', 'magenta', 'green', 'orange', 'blue', 'darkgreen', 'black', 'gray']
      fig = plt.figure(figsize=(5, 5))
      ax1 = fig.add_subplot(1, 1, 1)
      for i, (label, performance) in enumerate(best_performance[-1].items()):
          valid_budgets = [total_anchor[i+1]] * len(performance)
          ax1.plot(valid_budgets, performance, label=label, color=colors[i % len(colors)], marker='o')

      for budget_cut in total_anchor[1:-1]:
          ax1.axvline(x=budget_cut, color='black', linestyle='--')

      ax1.set_xlabel('Anchor size')
      ax1.set_ylabel('Performance')
      ax1.set_title('Performance vs. Anchor size for Successive Halving')

      ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      print(min(results['successive halving']))

    fig = plt.figure(figsize=(5, 5))
    i = 0
    for results in resultss:
        plt.plot(range(len(results['successive halving'])), results['successive halving'], color=colors[i], label='dataset'+str(i))
        plt.yscale('log')
        plt.legend(loc="upper right")
        i += 1
    plt.show()

if __name__ == '__main__':
    run(parse_args())
