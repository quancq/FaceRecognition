import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import utils, plot_utils


def calculate_class_distribution():
    org_dataset_dir = "/home/quanchu/Dataset/FaceImagCroppedWithAlignmentShorten"
    lst = []

    dirs = os.listdir(org_dataset_dir)
    for dir in dirs:
        full_dir_path = os.path.join(org_dataset_dir, dir)
        num_files = len(os.listdir(full_dir_path))

        lst.append((dir, num_files))
        # print("Class {} : {} files".format(dir, num_files))

    df = pd.DataFrame(lst, columns=["Class", "Number Samples"])

    save_dir = "./EDA"

    # Save file contain number samples of each class
    utils.save_csv(df, os.path.join(save_dir, "Class-NumSamples.csv"))
    print(df.head())

    # Save file contain statistic about class distribution
    arr = df["Number Samples"].values
    min, max, mean, std = arr.min(), arr.max(), arr.mean(), arr.std()
    stats = [("min", min), ("max", max), ("mean", mean), ("std", std)]

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

    print("In {}: has {} dirs".format(org_dataset_dir, len(dirs)))


if __name__ == "__main__":
    calculate_class_distribution()
