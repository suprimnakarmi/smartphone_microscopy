"""
This script is used to plot the precision-recall curve for a given model and sample type.

Usage:
    python utils/plot_pr_curve.py --base_dir /path/to/base/dir --model_type model_type --sample_type sample_type --fold fold

Arguments:
    base_dir: path to the base directory where the outputs are stored
    model_type: the model type for which the pr curve is to be plotted
    sample_type: the sample type for which the pr curve is to be plotted
    fold: the fold for which the pr curve is to be plotted
    save: whether to save the plot or not
"""

import os.path
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .metrics import calculate_precision_recall_f1


def plot_pr_curve(args: Namespace):
    base_dir = args.base_dir
    model_type = args.model_type
    sample_type = args.sample_type
    fold = args.fold

    gt_annotation_file = os.path.join(
        base_dir,
        f"cysts_dataset_all/{sample_type}/fold_{fold}/{sample_type}_coco_annos_val.json",
    )

    pred_annotation_file = os.path.join(
        base_dir, f"outputs/{sample_type}/{model_type}/fold_{fold}/results.bbox.json"
    )

    if not os.path.exists(pred_annotation_file):
        print(f"Predictions for {model_type} not found")
        return

    confidence_thresholds = np.arange(0.7, 0.0, -0.05)
    precisions = {
        0: np.zeros(len(confidence_thresholds)),
        1: np.zeros(len(confidence_thresholds)),
    }

    recalls = {
        0: np.zeros(len(confidence_thresholds)),
        1: np.zeros(len(confidence_thresholds)),
    }

    for i, conf_threshold in enumerate(confidence_thresholds):
        metrics_df = calculate_precision_recall_f1(
            pred_annotation_file, gt_annotation_file, conf_threshold
        )
        precisions[0][i] = metrics_df[metrics_df["category"] == "crypto"][0][
            "precision"
        ]
        precisions[1][i] = metrics_df[metrics_df["category"] == "giardia"][0][
            "precision"
        ]

        recalls[0][i] = metrics_df[metrics_df["category"] == "crypto"][0]["recall"]
        recalls[1][i] = metrics_df[metrics_df["category"] == "giardia"][0]["recall"]

    crypto_metrics_df = pd.DataFrame(
        {
            "precision": precisions[0],
            "recall": recalls[0],
        }
    )
    giardia_metrics_df = pd.DataFrame(
        {
            "precision": precisions[1],
            "recall": recalls[1],
        }
    )
    # plot giardia and crypto pr curves for the given sample type
    sns.lineplot(
        data=crypto_metrics_df, x="recall", y="precision", label="Cryptosporidium"
    )
    sns.lineplot(data=giardia_metrics_df, x="recall", y="precision", label="Giardia")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.title(
        f"Precision-Recall Curve for {sample_type.replace('_', ' ')} using {model_type.replace('_', ' ').capitalize()}"
    )
    if args.save:
        save_path = os.path.join(
            base_dir,
            f"outputs/{sample_type}/{model_type}/pr_curve.png",
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved pr curve at {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system/",
    )
    parser.add_argument("--model_type", type=str, default="retinanet")
    parser.add_argument("--sample_type", type=str, default="brightfield_sample")
    parser.add_argument(
        "--fold", type=int, default=1, help="Fold to use for evaluation"
    )
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    plot_pr_curve(args)
