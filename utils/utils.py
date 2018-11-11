import numpy as np
import pandas as pd
from datetime import datetime
import os
import shutil
from settings import DEFAULT_TIME_FORMAT
import json


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    return time.strftime(fmt)


def get_time_obj(time_str, fmt=DEFAULT_TIME_FORMAT):
    return datetime.strptime(time_str, fmt)


def transform_time_fmt(time_str, src_fmt, dst_fmt=DEFAULT_TIME_FORMAT):
    time_obj = get_time_obj(time_str, src_fmt)
    return get_time_str(time_obj, dst_fmt)


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_parent_dirs(path):
    dir = path[:path.rfind("/")]
    make_dirs(dir)


def get_dirs(parent_dir):
    dirs = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir)]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    return dirs


def get_paths(parent_dir):
    dirs = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir)]
    return dirs


def get_file_names(parent_dir):
    file_names = os.listdir(parent_dir)
    return file_names


def copy_file(src_path, dst_path):
    try:
        make_parent_dirs(dst_path)
        shutil.copyfile(src_path, dst_path)
        print("Copy file from {} to {} done".format(src_path, dst_path))
        return True
    except Exception:
        print("Error: when copy file from {} to {}".format(src_path, dst_path))
        return False


def copy_files(src_dst_paths):
    total_paths = len(src_dst_paths)
    num_success = 0
    for i, (src_path, dst_path) in enumerate(src_dst_paths):
        print("Copying {}/{} ...".format(i+1, total_paths))
        is_success = copy_file(src_path, dst_path)
        if is_success:
            num_success += 1

    print("Copy {}/{} files done".format(num_success, total_paths))
    return num_success


def save_json(data, path):
    make_parent_dirs(path)
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, default=MyEncoder)
    print("Save json data (size = {}) to {} done".format(len(data), path))


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_csv(df, path, fields=None, **kwargs):
    make_parent_dirs(path)
    if fields is not None:
        df = df[fields]
    df.to_csv(path, index=False, **kwargs)
    print("Save csv data (size = {}) to {} done".format(df.shape[0], path))


def load_csv(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    print("Load csv data (size = {}) from {} done".format(df.shape[0], path))

    return df
