import numpy as np
import pandas as pd
import os
from utils_dir import utils, plot_utils
import argparse


def calculate_class_distribution(dataset_dir, save_dir):
    # dataset_dir = "/home/quanchu/Dataset/FaceImagCroppedWithAlignmentShorten"
    lst = []

    dirs = os.listdir(dataset_dir)
    for dir in dirs:
        full_dir_path = os.path.join(dataset_dir, dir)
        num_files = len(os.listdir(full_dir_path))

        lst.append((dir, num_files))
        # print("Class {} : {} files".format(dir, num_files))

    df = pd.DataFrame(lst, columns=["Class", "Number Samples"])

    # save_dir = "./EDA/Version2"

    # Save file contain number samples of each class
    utils.save_csv(df, os.path.join(save_dir, "Class-NumSamples.csv"))
    print(df.head())

    # Save file contain statistic about class distribution
    arr = df["Number Samples"].values
    min, max, mean, std, sum = arr.min(), arr.max(), arr.mean(), arr.std(), arr.sum()
    stats = [("min", min), ("max", max), ("mean", mean), ("std", std), ("sum", sum)]

    stats_df = pd.DataFrame(stats, columns=["Statistic", "Value"])
    utils.save_csv(stats_df, os.path.join(save_dir, "Statistic.csv"))

    # Plot number samples of each class
    plot_utils.plot_histogram(
        arr.tolist(),
        num_bins=300,
        save_path=os.path.join(save_dir, "ClassDistribution.jpg"),
        color="C2",
        figsize=(15, 8),
        title="Class Distribution",
        xlabel="Number samples",
        ylabel="Number classes",
    )

    # print("In {}: has {} dirs".format(dataset_dir, len(dirs)))
    print("EDA:: Save calculate class distribution to {} done".format(save_dir))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--save_dir", required=True)

    args = vars(ap.parse_args())
    dataset_dir = args["dataset_dir"]
    save_dir = args["save_dir"]

    calculate_class_distribution(dataset_dir=dataset_dir, save_dir=save_dir)
    pass

