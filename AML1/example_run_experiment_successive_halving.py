import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from successive_halving import SuccessiveHalving


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()


def run(args):
    df = pd.read_csv(args.configurations_performance_file)
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    sh = SuccessiveHalving(config_space,args.max_anchor_size,halving_factor=2)
    surrogate_model_sh = SurrogateModel(config_space)
    surrogate_model_sh.fit(df)

    results = {
            'random_search': [1.0],
            'smbo': [1.0],
            'successive_halving': [1.0]
        }

    for idx in range(args.num_iterations):
        performance_sh = sh.run(surrogate_model_sh)
        results['successive_halving'].append(min(results['successive_halving'][-1], performance_sh))
    colors = ['red', 'magenta', 'green', 'orange', 'blue', 'darkgreen', 'black', 'gray']
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    for i, (label, performance) in enumerate(sh.best_performance[-1].items()):
        valid_budgets = [sh.total_anchor[i+1]] * len(performance)
        ax1.plot(valid_budgets, performance, label=label, color=colors[i % len(colors)], marker='o')

    for budget_cut in sh.total_anchor[1:-1]:
        ax1.axvline(x=budget_cut, color='black', linestyle='--')

    ax1.set_xlabel('Anchor size')
    ax1.set_ylabel('Performance')
    ax1.set_title('Performance vs. Anchor size for Successive Halving')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    print(min(results['successive_halving']))
    plt.show()

if __name__ == '__main__':
    run(parse_args())
