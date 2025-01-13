def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='/content/drive/MyDrive/lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='/content/drive/MyDrive/lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5)

    return parser.parse_known_args()[0]

def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    
    # external SurrogateModel pretrained
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    smbo = SequentialModelBasedOptimization(config_space)
    initial_runs = []
    configs = [dict(config_space.sample_configuration()) for _ in range(100)]
    for config in configs:
        config['anchor_size'] = args.max_anchor_size
        config['score'] = surrogate_model.predict(config)
    
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
    
    results = {
        'BO': [1.0]
        }
    
    for idx in range(args.num_iterations):
        smbo.fit_model()
        theta_new = smbo.select_configuration()
        
        theta_new['anchor_size'] = args.max_anchor_size
        theta_new_ = theta_new.copy()
        performance = surrogate_model.predict(theta_new)

        theta_new_ = theta_new_.pop('anchor_size')
        new_best = performance
        results['BO'].append(new_best)
    
        smbo.update_runs((theta_new_, performance))



    plt.plot(range(len(results['BO'])), results['BO'])
    plt.yscale('log')
    plt.show()