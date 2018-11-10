from utils import utils
import numpy as np
import time
import face_recognition


def get_face_encodings(dir="../Dataset/Temp"):
    start_time = time.time()

    file_paths = utils.get_paths(dir)
    print("In dir {} has {} files".format(dir, len(file_paths)))

    # Load image and calculate face encoding from file paths
    images = []
    face_encodings = []
    available_file_paths = []
    result = []
    error_load_files = []
    error_calc_encoding = []

    for file_path in file_paths:
        try:
            image = face_recognition.load_image_file(file_path)

            # Calculate face encoding of each image
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                face_encodings.append(face_encoding)
                available_file_paths.append(file_path)
                result.append((file_path, face_encoding))

            except IndexError:
                error_calc_encoding.append(file_path)
                # print("Error: Can not locate any faces in ", file_path)

        except Exception:
            error_load_files.append(file_path)
            # print("Error: Can not load image from ", file_path)

    exec_time = time.time() - start_time
    print("In dir {}: Calculate face encoding successfully {}/{} files. Time : {:.2f} seconds".format(
        dir, len(available_file_paths), len(file_paths), exec_time))
    if len(error_load_files) > 0:
        print("{} files can not load image: {}".format(len(error_load_files), error_load_files))
    if len(error_calc_encoding) > 0:
        print("{} files can not calculate face encoding: {}".format(
            len(error_calc_encoding), error_calc_encoding))

    return result


def get_sorted_similarity_images(dir="../Dataset/Temp"):
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


if __name__ == "__main__":
    dir = "../Dataset/CroppedWithAlignedSamples/m.01_0d4"
    # dir = "../Dataset/Temp"

    sorted_similarity = get_sorted_similarity_images(dir)

    for file_path, similarity in sorted_similarity:
        print("File {}: similarity : {}".format(file_path, similarity))
