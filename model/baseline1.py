from utils import utils
import numpy as np
import time
import os
from collections import defaultdict
import face_recognition


def get_face_encodings(dir="../Temp/Dataset"):
    '''
    Calculate face encodings of each images in directory
    :param dir: Directory contain images need calculate face encoding
    :return: list of tuple that contain file name and face encoding of file
    '''
    start_time = time.time()

    file_names = utils.get_file_names(dir)
    print("Start calculate face encoding of dir {} which has {} files".format(dir, len(file_names)))

    # Load image and calculate face encoding from file paths
    face_encodings = []
    available_file_names = []
    result = []
    error_load_files = []
    error_calc_encoding = []

    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        try:
            image = face_recognition.load_image_file(file_path)

            # Calculate face encoding of each image
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                face_encodings.append(face_encoding)
                available_file_names.append(file_name)
                result.append((file_name, face_encoding.tolist()))

            except IndexError:
                error_calc_encoding.append(file_path)
                # print("Error: Can not locate any faces in ", file_path)

        except Exception:
            error_load_files.append(file_path)
            # print("Error: Can not load image from ", file_path)

    exec_time = time.time() - start_time
    print("In dir {}: Calculate face encoding successfully {}/{} files. Time : {:.2f} seconds".format(
        dir, len(available_file_names), len(file_names), exec_time))
    if len(error_load_files) > 0:
        print("{} files can not load image: {}".format(len(error_load_files), error_load_files))
    if len(error_calc_encoding) > 0:
        print("{} files can not calculate face encoding: {}".format(
            len(error_calc_encoding), error_calc_encoding))

    return result


def get_sorted_similarity_images(dir="../Temp/Dataset"):
    start_time = time.time()
    # Calculate face encoding of each images in directory
    file_face_encodings = get_face_encodings(dir)
    file_paths, face_encodings = [], []
    for fpath, fencoding in file_face_encodings:
        file_paths.append(fpath)
        face_encodings.append(fencoding)
    num_files = len(file_paths)

    similarities = []

    for i, fencoding in enumerate(face_encodings):
        other_fencodings = face_encodings[:i] + face_encodings[i+1:]
        similarity = face_recognition.compare_faces(other_fencodings, fencoding)
        similarity = np.sum(similarity) / num_files
        similarities.append((file_paths[i], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    exec_time = time.time() - start_time
    print("Get sorted similarity images from {} done. Time : {:.2f} seconds".format(dir, exec_time))

    return similarities


def save_face_encoding(dataset_dir="../Temp/Dataset", save_path="../Temp/Dataset/face_encodings_v1.json"):
    start_time = time.time()
    map = {}
    total_files = 0

    dirs = utils.get_file_names(parent_dir=dataset_dir)
    total_dirs = len(dirs)
    for i, dir in enumerate(dirs):
        fencoding_of_dir = get_face_encodings(os.path.join(dataset_dir, dir))
        fencoding_map = {fname: fencoding for fname, fencoding in fencoding_of_dir}
        total_files += len(fencoding_map)

        map.update({dir: fencoding_map})
        print("\nCalculate {}/{} face encoding dir done".format(i+1, total_dirs))

    exec_time = time.time() - start_time
    print("\nCalculate face encodings of {} dirs and {} files in dir {} done. Time : {:.2f} seconds".format(
        len(map), total_files, dataset_dir, exec_time))

    utils.save_json(map, save_path)
    print("Save face encoding (size = {}) done".format(len(map)))


if __name__ == "__main__":
    # dir = "../Dataset/CroppedWithAlignedSamples/m.01_0d4"
    dir = "../Dataset/CroppedWithAlignedSamples"

    save_face_encoding(dataset_dir=dir, save_path=os.path.join(dir, "face_encodings_v1.json"))
