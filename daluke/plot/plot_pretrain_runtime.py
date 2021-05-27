import os

import numpy as np
import matplotlib.pyplot as plt

from daluke.analysis.pretrain import TrainResults
from daluke.pretrain.train import Hyperparams


data = [
    {
        "colour": "tab:orange",
        "label": "V100 32 GB",
        "gpus": [1, 2, 3, 4],
        "locs": [
            "johnny-juliet-v1",
            "johnny-juliet-v2",
            "johnny-juliet-v3",
            "johnny-juliet-v4",
        ],
    },
    {
        "colour": "tab:red",
        "label": "V100 32 GB",
        "gpus": [1, 2, 3, 4],
        "locs": [
            "johnny-juliet-v32-1",
            "johnny-juliet-v32-2",
            "johnny-juliet-v32-3",
            "johnny-juliet-v32-4",
        ],
    },
    {
        "colour": "tab:blue",
        "label": "A100 40 GB",
        "gpus": [1, 2],
        "locs": [
            "johnny-juliet-1",
            "johnny-juliet-2",
        ]
    },
    {
        "colour": "blue",
        "label": "A100 40 GB",
        "gpus": [1, 2],
        "locs": [
            "johnny-juliet-a32-1",
            "johnny-juliet-a32-2",
        ]
    }
]
Hyperparams.ignore_fp16_cuda_access = True


def run():
    for i, dat in enumerate(data):
        runtimes = list()
        for loc in dat["locs"]:
            TrainResults.subfolder = loc
            res = TrainResults.load("/work3/s183912/pdata2")
            runtimes.append(res.runtime[1].sum())
        baseline = runtimes[0] * dat["gpus"][0]
        legend = { "label": "Optimal scaling" } if i == 0 else dict()
        plt.plot(dat["gpus"], [baseline / gpus for gpus in dat["gpus"]], marker=".", ms=10, color="darkgrey", ls="-.", **legend)
        Hyperparams.subfolder = loc
        params = Hyperparams.load("/work3/s183912/pdata2")
        label = dat["label"]
        if params.fp16:
            label += " with AMP"
        label += ", SB%i" % params.ff_size
        plt.plot(dat["gpus"], runtimes, marker=".", ms=12, color=dat["colour"], label=label)
    plt.grid()
    plt.title("Pretraining Time of One Epoch")
    plt.xlabel("Number of GPU's")
    plt.ylabel("Training Time per Epoch [s]")
    plt.xticks([1, 2, 3, 4])
    plt.ylim(bottom=0)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("/work3/s183912/pdata2/runtime.png")
    plt.close()

if __name__ == "__main__":
    run()
