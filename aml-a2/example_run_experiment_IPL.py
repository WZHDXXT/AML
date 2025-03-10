import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
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
    parser.add_argument("--minimal_anchor", type=int, default=724)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=100)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    ipl = IPL(surrogate_model, args.minimal_anchor, args.max_anchor_size)
    best_so_far = None
    accumulated_anchor_size = 0
    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = ipl.evaluate_model(best_so_far, theta_new)
        for anchor, _ in result:
            accumulated_anchor_size += anchor
        final_result = result[-1][1]
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        plt.plot(x_values, y_values, "-o")

    plt.title(
        "Best Performance:{:.4f}, Accumulated Anchor Size:{}".format(
            best_so_far, accumulated_anchor_size
        )
    )
    plt.show()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
