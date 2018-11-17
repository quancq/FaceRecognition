from utils import utils
import os
from sklearn.model_selection import train_test_split
import argparse


def split_dataset(src_dataset_dir, dst_dataset_dir, test_size=0.1):
    img_names, labels = [], []
    for dir in utils.get_dir_names(src_dataset_dir):
        for name in utils.get_file_names(os.path.join(src_dataset_dir, dir)):
            img_names.append(name)
            labels.append(dir)

    train_fnames, test_fnames, train_labels, test_labels = train_test_split(
        img_names, labels, test_size=test_size, stratify=labels, random_state=7)

    train_dir = os.path.join(dst_dataset_dir, "Train")
    test_dir = os.path.join(dst_dataset_dir, "Test")

    # Save new split dataset
    src_dst_paths = []
    for i, dir in enumerate([train_dir, test_dir]):
        fnames = train_fnames if i == 0 else test_fnames
        labels = train_labels if i == 0 else test_labels
        for fname, label in zip(fnames, labels):
            src_path = os.path.join(src_dataset_dir, label, fname)
            dst_path = os.path.join(dir, label, fname)
            src_dst_paths.append((src_path, dst_path))

    utils.copy_files(src_dst_paths)

    print("Split dataset from {} (size = {}) to dir {} (size = {}) and dir {} (size = {})".format(
        src_dataset_dir, len(labels), train_dir, len(train_labels), test_dir, len(test_labels)))


if __name__ == "__main__":
    # src_dataset_dir = "../Dataset/CroppedWithAlignedSamples"
    # dst_dataset_dir = "../Dataset/Train_Test1"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset_dir", required=True, help="Directory path of source dataset contain multi folder that each folder represent unique person")
    ap.add_argument("--dst_dataset_dir", required=True, help="Directory path to save split dataset")
    ap.add_argument("--test_size", help="Test size of test dataset", default=0.1)

    args = vars(ap.parse_args())
    src_dataset_dir = args["src_dataset_dir"]
    dst_dataset_dir = args["dst_dataset_dir"]
    test_size = int(args["test_size"])

    split_dataset(src_dataset_dir, dst_dataset_dir, test_size=test_size)
