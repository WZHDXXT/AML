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
    parser.add_argument("--minimal_anchor", type=int, default=724)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=100)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    lccv = LCCV(surrogate_model, args.minimal_anchor, args.max_anchor_size)
    ipl = IPL(surrogate_model, args.minimal_anchor, args.max_anchor_size)
    best_so_far_lccv = None
    best_so_far_ipl = None
    best_so_far_random = None
    accumulated_anchor_size_lccv = 0
    accumulated_anchor_size_ipl = 0
    accumulated_anchor_size_random = 0
    results = {"LCCV": [], "IPL": [], "Random_Search": []}
    x_values_lccv = []
    y_values_lccv = []
    x_values_ipl = []
    y_values_ipl = []

    for idx in range(args.num_iterations):

        theta_new = dict(config_space.sample_configuration())
        # run LCCV
        xy_values_lccv = []
        result_lccv = lccv.evaluate_model(best_so_far_lccv, theta_new)
        for anchor, _ in result_lccv:
            accumulated_anchor_size_lccv += anchor

        final_result_lccv = result_lccv[-1][1]
        if best_so_far_lccv is None or final_result_lccv < best_so_far_lccv:
            best_so_far_lccv = final_result_lccv
        x_values_lccv.append([i[0] for i in result_lccv])
        y_values_lccv.append([i[1] for i in result_lccv])
        results["LCCV"].append(best_so_far_lccv)

        # plt.plot(x_values_lccv, y_values_lccv, "-o")
        xy_values_lccv.append((x_values_lccv, y_values_lccv))

        # run IPL
        xy_values_ipl = []
        result_ipl = ipl.evaluate_model(best_so_far_ipl, theta_new)
        for anchor, _ in result_ipl:
            accumulated_anchor_size_ipl += anchor
        final_result_ipl = result_ipl[-1][1]
        if best_so_far_ipl is None or final_result_ipl < best_so_far_ipl:
            best_so_far_ipl = final_result_ipl
        x_values_ipl.append([i[0] for i in result_ipl])
        y_values_ipl.append([i[1] for i in result_ipl])
        results["IPL"].append(best_so_far_ipl)
        # plt.plot(x_values_ipl, y_values_ipl, "-o")
        xy_values_ipl.append((x_values_ipl, y_values_ipl))

        # run random search

        theta_new["anchor_size"] = args.max_anchor_size
        performance_random = surrogate_model.predict(theta_new=theta_new)
        accumulated_anchor_size_random += args.max_anchor_size
        if best_so_far_random is None or performance_random < best_so_far_random:
            best_so_far_random = performance_random
        results["Random_Search"].append(best_so_far_random)


        
    plt.figure()
    for x_values, y_values in xy_values_lccv:
        for i in range(len(x_values)):
            plt.plot(x_values[i], y_values[i], "-o")
    plt.title(
        "LCCV\nBest Performance:{:.4f}, Accumulated Anchor Size:{}".format(
            best_so_far_lccv, accumulated_anchor_size_lccv
        )
    )
    plt.xlabel("Anchor Size")
    plt.ylabel("Performance")
    plt.show()

    plt.figure()
    for x_values, y_values in xy_values_ipl:
        for i in range(len(x_values)):
            plt.plot(x_values[i], y_values[i], "-o")
    plt.title(
        "IPL\nBest Performance:{:.4f}, Accumulated Anchor Size:{}".format(
            best_so_far_ipl, accumulated_anchor_size_ipl
        )
    )
    plt.xlabel("Anchor Size")
    plt.ylabel("Performance")
    plt.show()

    plt.figure()
    plt.plot(
        range(len(results["LCCV"])),
        results["LCCV"],
        label="LCCV",
    )
    plt.plot(
        range(len(results["IPL"])),
        results["IPL"],
        label="IPL",
    )
    plt.plot(
        range(len(results["Random_Search"])),
        results["Random_Search"],
        label="Random Search",
    )
    plt.title(
        "Comparison of Performance\n LCCV Best ={:.4f}, IPL Best ={:.4f}, Random Best ={:.4f}".format(
            best_so_far_lccv, best_so_far_ipl, best_so_far_random
        )
    )
    plt.xlabel("Iterations")
    plt.ylabel("Performance")
    plt.legend()
    plt.show()

    x_labels = ["LCCV", "IPL", "Random Search"]
    x_positions = range(len(x_labels))  # [0, 1, 2]
    y_values = [
        accumulated_anchor_size_lccv,
        accumulated_anchor_size_ipl,
        accumulated_anchor_size_random,
    ]

    plt.figure()
    plt.bar(x_positions, y_values, color=["blue", "green", "orange"])
    plt.title(
        "Comparison of Accumulated Anchor Size\n LCCV={}, IPL={}, Random={}".format(
            accumulated_anchor_size_lccv,
            accumulated_anchor_size_ipl,
            accumulated_anchor_size_random,
        )
    )
    plt.xticks(x_positions, x_labels)
    plt.ylabel("Accumulated Anchor Size")
    plt.show()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
