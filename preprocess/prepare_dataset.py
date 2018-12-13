from utils_dir import utils
import os
from sklearn.model_selection import train_test_split
import argparse
import math
import random
import time


def split_dataset(src_dataset_dir, dst_dataset_dir, test_size=0.1, valid_size=0.0):
    start_time = time.time()

    train, valid, test = [], [], []
    for dir in utils.get_dir_names(src_dataset_dir):
        fnames = utils.get_file_names(os.path.join(src_dataset_dir, dir))
        num_fnames = len(fnames)
        num_test = int(math.ceil(test_size * num_fnames))
        num_valid = int(math.ceil(valid_size * num_fnames))
        num_train = num_fnames - num_test - num_valid

        random.shuffle(fnames)
        dir_fnames = [(dir, fname) for fname in fnames]
        train.extend(dir_fnames[:num_train])
        valid.extend(dir_fnames[num_train: num_train+num_valid])
        test.extend(dir_fnames[-num_test:])

    train_dir = os.path.join(dst_dataset_dir, "Train")
    valid_dir = os.path.join(dst_dataset_dir, "Valid")
    test_dir = os.path.join(dst_dataset_dir, "Test")

    # Save new split dataset
    lst = [(train_dir, train), (valid_dir, valid), (test_dir, test)]
    src_dst_paths = []
    for dst_parent_dir, dir_fnames in lst:
        for dir, fname in dir_fnames:
            src_path = os.path.join(src_dataset_dir, dir, fname)
            dst_path = os.path.join(dst_parent_dir, dir, fname)
            src_dst_paths.append((src_path, dst_path))

    utils.copy_files(src_dst_paths)

    exec_time = time.time() - start_time

    print("\nSplit dataset from {} (size = {}) to :".format(src_dataset_dir, len(src_dst_paths)))
    print("---- {} (size = {})".format(train_dir, len(train)))
    print("---- {} (size = {})".format(valid_dir, len(valid)))
    print("---- {} (size = {})".format(test_dir, len(test)))
    print("Time : {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    # src_dataset_dir = "../Dataset/CroppedWithAlignedSamples"
    # dst_dataset_dir = "../Dataset/Train_Test1"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset_dir", required=True,
                    help="Directory path of source dataset contain multi folder that each folder represent unique person")
    ap.add_argument("--dst_dataset_dir", required=True, help="Directory path to save split dataset")
    ap.add_argument("--valid_size", help="Valid size of test dataset", default=0.15)
    ap.add_argument("--test_size", help="Test size of test dataset", default=0.15)

    args = vars(ap.parse_args())
    src_dataset_dir = args["src_dataset_dir"]
    dst_dataset_dir = args["dst_dataset_dir"]
    valid_size = float(args["valid_size"])
    test_size = float(args["test_size"])

    split_dataset(src_dataset_dir, dst_dataset_dir, valid_size=valid_size, test_size=test_size)
