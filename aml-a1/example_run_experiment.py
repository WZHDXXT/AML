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
    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space)
    sh = SuccessiveHalving(config_space,args.max_anchor_size,halving_factor=2)


    surrogate_model_rs = SurrogateModel(config_space)
    surrogate_model_smbo = SurrogateModel(config_space)
    surrogate_model_sh = SurrogateModel(config_space)
    surrogate_model_rs.fit(df)
    surrogate_model_smbo.fit(df)
    surrogate_model_sh.fit(df)

    configs = [dict(config_space.sample_configuration()) for _ in range(200)]
    for config in configs:
        config['anchor_size'] = args.max_anchor_size
        config['score'] = surrogate_model_smbo.predict(config)
    
    # configs_ = configs.copy()
    # for c in configs_:
    #     c = c.pop('anchor_size')
    # variables = list(df.keys())
    # variables.remove('anchor_size')
    # df_config = pd.DataFrame([[i[j] for j in variables] for i in configs_], columns=variables)
    # m, n = df_config.shape
    # initial_runs = []
    # for _, row in df_config.iterrows():
    #     config = dict(row.head(n-1))
    #     p = row['score']
    #     initial_runs.append((config, p))
    
    df_config = pd.DataFrame([{k: v for k, v in config.items() if k != 'anchor_size'} for config in configs])
    initial_runs = [(row.drop('score').to_dict(), row['score']) for _, row in df_config.iterrows()]
    smbo.initialize(initial_runs)

    results = {
            'random search': [1.0],
            'smbo': [1.0],
            'successive halving': [1.0]
        }

    for idx in range(args.num_iterations):
        theta_new_rs = dict(random_search.select_configuration())
        theta_new_rs['anchor_size'] = args.max_anchor_size
        performance_rs = surrogate_model_rs.predict(theta_new_rs)
        # ensure to only record improvements
        results['random search'].append(min(results['random search'][-1], performance_rs))
        random_search.update_runs((theta_new_rs, performance_rs))

        theta_new_smbo = dict(smbo.select_configuration())
        theta_new_smbo['anchor_size'] = args.max_anchor_size
        theta_new_smbo_update = theta_new_smbo.copy()
        performance_smbo = surrogate_model_smbo.predict(theta_new_smbo)
        results['smbo'].append(min(results['smbo'][-1], performance_smbo))
        del theta_new_smbo_update['anchor_size']
        smbo.update_runs((theta_new_smbo_update, performance_smbo))

        performance_sh = sh.run(surrogate_model_sh)
        results['successive halving'].append(min(results['successive halving'][-1], performance_sh))
    print('random search:', min(results['random search']))
    print('SMBO', min(results['smbo']))
    print('successive halving', min(results['successive halving']))

    plt.plot(range(len(results['random search'])), results['random search'], label='random search')
    plt.plot(range(len(results['smbo'])), results['smbo'], label='smbo')
    plt.plot(range(len(results['successive halving'])), results['successive halving'], label='successive halving')
    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    run(parse_args())
