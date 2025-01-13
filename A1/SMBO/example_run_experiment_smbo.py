import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from smbo import SequentialModelBasedOptimization
from surrogate_model import SurrogateModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5)

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
    i = 0
    for df in df_list:
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

      print(min(results_BO['BO']))
      plt.plot(range(len(results_BO['BO'])), results_BO['BO'], color=colors[i], label='dataset'+str(i))
      plt.yscale('log')
      i += 1
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    run(parse_args())