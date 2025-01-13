import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lccv import LCCV
from ipl import IPL
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )
    parser.add_argument(
        "--configurations_performance_file",
        type=str,
        default="config_performances_dataset-6.csv",
    )
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument("--minimal_anchor", type=int, default=256)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=5)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model_lccv = SurrogateModel(config_space)
    surrogate_model_ipl = SurrogateModel(config_space)
    surrogate_model_lccv.fit(df)
    surrogate_model_ipl.fit(df)
    
    lccv = LCCV(surrogate_model_lccv, args.minimal_anchor, args.max_anchor_size)
    ipl = IPL(surrogate_model_ipl, args.minimal_anchor, args.max_anchor_size)
    
    best_so_far_lccv = None
    best_so_far_ipl = None
    best_lccv = []
    best_ipl = []
    
    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = lccv.evaluate_model(best_so_far_lccv, theta_new)
        final_result = result[-1][1]
        if best_so_far_lccv is None or final_result < best_so_far_lccv:
            best_so_far_lccv = final_result
        best_lccv.append(best_so_far_lccv)
        
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        # plt.plot(x_values, y_values, "-o")
    
    plt.plot(range(len(best_lccv)), best_lccv)

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = ipl.evaluate_model(best_so_far_ipl, theta_new)
        final_result = result[-1][1]
        if best_so_far_ipl is None or final_result < best_so_far_ipl:
            best_so_far_ipl = final_result
        best_ipl.append(best_so_far_ipl)
        
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        # plt.plot(x_values, y_values, "-o")
    
    plt.plot(range(len(best_ipl)), best_ipl)


    plt.show()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
