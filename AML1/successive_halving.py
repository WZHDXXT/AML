import ConfigSpace
import numpy as np
import pandas as pd

class SuccessiveHalving(object):
    def __init__(self, config_space: ConfigSpace.ConfigurationSpace, max_anchor_size, halving_factor: int = 2):
        self.config_space = config_space
        self.halving_factor = halving_factor
        self.max_anchor_size = max_anchor_size
        self.best_performance = []
        self.best = np.inf
        self.total_anchor = [0]

    def run(self, surrogate_model):
        total_performance = {}
        current_anchor = 25
        configs = [dict(self.config_space.sample_configuration()) for _ in range(64)]
        while(current_anchor<=self.max_anchor_size):
            self.total_anchor.append(current_anchor)
            performances = []
            m = len(configs)

            for config in configs:
                for hp in list(self.config_space.values()):
                    if hp.name not in config.keys():
                        config[hp.name] = hp.default_value
                config['anchor_size'] = current_anchor

                performance = surrogate_model.predict(config)
                performances.append(performance)

            total_performance['M'+str(current_anchor)] = performances
            performances = np.array(performances)
            performances_sorted = np.argsort(performances)
            num = int(m/self.halving_factor)

            configs = [configs[index] for index in performances_sorted[:num]]
            current_anchor *= self.halving_factor

        p = total_performance.get(list(total_performance.keys())[-1])[0]
        if p < self.best:
            self.best = p
            self.best_performance.append(total_performance)
        return self.best