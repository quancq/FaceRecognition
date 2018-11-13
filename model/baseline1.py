from utils import utils
from utils import project_utils
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math
import argparse
from collections import defaultdict
import face_recognition
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


def _get_face_encodings(dir="../Temp/Dataset"):
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


def get_face_encodings(image_path):
    '''
    Calculate face encodings of each images
    :param image_path: path of images need calculate face encoding
    :return: list (size = 128) of face encodings or empty list if occur exception
    '''

    start_time = time.time()
    face_encoding = []
    try:
        image = face_recognition.load_image_file(image_path)

        # Calculate face encoding of each image
        try:
            face_encoding = face_recognition.face_encodings(image)[0].tolist()

            exec_time = time.time() - start_time
            print("Calculate face encoding of {} done. Time : {:.2f} seconds".format(image_path, exec_time))

        except IndexError:
            print("Error: Can not locate any faces in ", image_path)

    except Exception:
        print("Error: Can not load image from ", image_path)
        
    return face_encoding


def calculate_similarity(face_encodings):
    num_faces = len(face_encodings)
    similarities = []
    for i, fencoding in enumerate(face_encodings):
        other_fencodings = face_encodings[:i] + face_encodings[i+1:]
        similarity = face_recognition.compare_faces(other_fencodings, fencoding)
        similarity = np.sum(similarity) / num_faces
        similarities.append(similarity)

    return similarities


def get_sorted_similarity_images(dir="../Temp/Dataset"):
    start_time = time.time()
    # Calculate face encoding of each images in directory
    file_face_encodings = _get_face_encodings(dir)
    file_names, face_encodings = [], []
    for fname, fencoding in file_face_encodings:
        file_names.append(fname)
        face_encodings.append(np.array(fencoding))

    similarities = [(fname, sim) for fname, sim in zip(file_names, calculate_similarity(face_encodings))]

    # num_files = len(file_names)
    # for i, fencoding in enumerate(face_encodings):
    #     other_fencodings = face_encodings[:i] + face_encodings[i+1:]
    #     similarity = face_recognition.compare_faces(other_fencodings, fencoding)
    #     similarity = np.sum(similarity) / num_files
    #     similarities.append((file_names[i], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    exec_time = time.time() - start_time
    print("Get sorted similarity images from {} done. Time : {:.2f} seconds".format(dir, exec_time))

    return similarities


def load_face_encoding(face_encoding_dir, file_names=[], all_files=False):
    face_encoding_map = {}
    if all_files:
        file_names = utils.get_file_names(face_encoding_dir)

    for file_name in file_names:
        map = utils.load_json(os.path.join(face_encoding_dir, file_name))
        face_encoding_map.update({file_name: map})

    return face_encoding_map


def save_face_encoding(dataset_dir="../Temp/Dataset/Original", save_dir="../Temp/Dataset/Process"):
    start_time = time.time()
    save_dir = os.path.join(save_dir, "face_encodings")
    total_files = 0

    dirs = utils.get_dir_names(parent_dir=dataset_dir)
    total_dirs = len(dirs)
    for i, dir in enumerate(dirs):
        fencoding_of_dir = _get_face_encodings(os.path.join(dataset_dir, dir))
        fencoding_map = {fname: fencoding for fname, fencoding in fencoding_of_dir}
        total_files += len(fencoding_map)

        save_path = os.path.join(save_dir, dir)
        utils.save_json(fencoding_map, save_path)
        print("Calculate and Save {}/{} face encoding dir done".format(i+1, total_dirs))

    exec_time = time.time() - start_time
    print("\nCalculate face encodings of {} dirs and {} files in dir {} done".format(
        total_dirs, total_files, dataset_dir))
    print("Save face encoding to dir {} done".format(save_dir))
    print("Time : {:.2f} seconds".format(exec_time))


def create_subset_data(face_encoding_dir, src_dataset_dir, dst_dataset_dir):
    src_dst_copy_paths = []
    mids = utils.get_dir_names(face_encoding_dir)
    for i, mid in enumerate(mids):
        file_path = os.path.join(face_encoding_dir, mid)
        img_fencoding_map = utils.load_json(file_path)

        file_names, face_encodings = [], []
        for fname, fencoding in img_fencoding_map.items():
            file_names.append(fname)
            face_encodings.append(np.array(fencoding))

        similarites = [(fname, sim) for fname, sim in zip(file_names, calculate_similarity(face_encodings))]
        similarites.sort(key=lambda x: x[1], reverse=True)

        print("{}/{} Calculate similarity of mid {} done".format(i+1, len(mids), mid))

        # Remain number images corresponding with highest similarity
        if len(similarites) > 0:
            num_remain_images = math.ceil(similarites[0][1] * 100)
            for fname, _ in similarites[:num_remain_images]:
                src_path = os.path.join(src_dataset_dir, mid, fname)
                dst_path = os.path.join(dst_dataset_dir, mid, fname)

                src_dst_copy_paths.append((src_path, dst_path))
        else:
            utils.make_dirs(os.path.join(dst_dataset_dir, mid))

    num_success = utils.copy_files(src_dst_copy_paths)
    print("Create subset data (size = {}) from {} to {} done".format(
        num_success, src_dataset_dir, dst_dataset_dir))


class BaseLine1Model:
    def __init__(self, training_data_dir, face_encoding_dir, mid_name_path):
        self.class_name = self.__class__.__name__

        self.training_data_dir = training_data_dir
        # self.valid_data_dir = valid_data_dir
        self.face_encoding_dir = face_encoding_dir
        self.mid_name_path = mid_name_path

        self.models = {}
        self.face_encoding_map = {}

        self._init_data()

    def _init_data(self):
        self.mid_name_map = project_utils.load_mid_name_map(self.mid_name_path)

        mids_train = utils.get_dir_names(self.training_data_dir)
        self.mid_class_map, self.class_mid_map = {}, {}
        for i, mid in enumerate(mids_train):
            self.mid_class_map.update({mid: i})
            self.class_mid_map.update({i: mid})

        self.num_classes = len(mids_train)

    def _load_face_encodings(self):
        start_time = time.time()
        X_train, y_train = [], []
        idx_fname_map = {}
        mids_train = utils.get_dir_names(self.training_data_dir)
        for mid in mids_train:
            self.face_encoding_map.update({mid: {}})

            mid_dir = os.path.join(self.training_data_dir, mid)
            file_names = utils.get_file_names(mid_dir)

            fencoding_map_of_mid = load_face_encoding(self.face_encoding_dir, file_names=[mid]).get(mid)
            # print("face_encoding_map of {} : {}".format(mid, list(fencoding_map_of_mid.keys())))
            num_calculated_files = 0
            for file_name in file_names:
                fencoding = fencoding_map_of_mid.get(file_name)
                if fencoding is None:
                    # Calculate face encoding of this file name
                    fencoding = get_face_encodings(image_path=os.path.join(mid_dir, file_name))
                    if len(fencoding) > 0:
                        num_calculated_files += 1
                        fencoding_map_of_mid.update({file_name: fencoding})

                if len(fencoding) > 0:
                    self.face_encoding_map[mid].update({file_name: fencoding})
                    idx_fname_map.update({len(X_train): (mid, file_name)})
                    X_train.append(fencoding)
                    y_train.append(self.mid_class_map.get(mid))

            # Save face encoding if there is any encoding just calculated
            if num_calculated_files > 0:
                utils.save_json(fencoding_map_of_mid, os.path.join(self.face_encoding_dir, mid))

        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        self.idx_fname_map = idx_fname_map

        exec_time = time.time() - start_time
        print("{}:: Load face encoding done. Time : {:.2f} seconds".format(self.class_name, exec_time))

    def train(self):
        start_train_time = time.time()
        print("{}:: Start train {} models : {} ...".format(
            self.class_name, len(self.models), list(self.models.keys())))

        if len(self.models) == 0:
            print("{}:: Can not train models because of empty models".format(self.class_name))
            return 0

        # Load face encoding which is calculated and save face encoding of images have not been calculated
        # Calculate X_train, y_train conform to sklearn's api
        self._load_face_encodings()

        print("{}:: Training data size : {}. Num classes : {}".format(
            self.class_name, self.X_train.shape[0], self.num_classes))
        print("{}:: X_train shape : {}, y_train shape : {}".format(
            self.class_name, self.X_train.shape, self.y_train.shape))
        # print("y_train: ", self.y_train)

        for model_name, model in self.models.items():
            t0 = time.time()
            model.fit(self.X_train, self.y_train)

            t1 = time.time()

            print("{}:: Training model {} is done. Time : {:.2f} seconds".format(
                self.class_name, model_name, t1-t0))
            break

        exec_train_time = time.time() - start_train_time
        print("{}:: Train {} models done. Time : {:.2f} seconds".format(
            self.class_name, len(self.models), exec_train_time))

        # Show training result
        # self.show_training_result()
        # print("{}:: Evaluate model on training data".format(self.class_name))
        # self.evaluate(self.X_train, self.y_train)

    def show_training_result(self):
        print("{}:: Show training result ... Not implement !".format(self.class_name))
        pass

    def predict_from_image_paths(self, predict_image_paths):
        face_encodings = []
        available_fnames, error_fnames = [], []
        fencoding_time = 0
        for img_path in predict_image_paths:
            fname = utils.get_fname_of_path(img_path)

            t0 = time.time()
            fencoding = get_face_encodings(image_path=img_path)
            t = time.time() - t0
            fencoding_time += t

            if len(fencoding) > 0:
                face_encodings.append(fencoding)
                available_fnames.append(fname)
            else:
                error_fnames.append(fname)

        X_pred = np.array(face_encodings)
        pred_class_id_df, pred_label_df, pred_times = self.predict(X_pred)
        pred_label_df["Image"] = available_fnames
        new_pred_time = {}
        for model_name, pred_time in pred_times:
            new_pred_time.update({model_name: pred_time + fencoding_time})

        print("{}:: Predict result ".format(self.class_name))
        print(pred_label_df.head())

        return pred_class_id_df, pred_label_df, new_pred_time

    def predict(self, X_pred):
        start_time = time.time()
        print("{}:: Start predict {} samples".format(self.class_name, X_pred.shape[0]))
        preds = {}
        pred_time = {}
        for model_name, model in self.models.items():
            t0 = time.time()
            y_pred = model.predict(X_pred)
            t = time.time() - t0

            pred_time.update({model_name: t})
            preds.update({model_name: y_pred})

            print("{}:: Model {} predict done. Time : {:.2f} seconds".format(
                self.class_name, model_name, t))
            break

        pred_class_id_df = pd.DataFrame(preds)

        num_models = len(self.models)
        # Calculate predict of ensemble
        preds = pred_class_id_df.values.tolist()
        major_preds, pred_prob = []
        for pred in preds:
            major_label, num_models_pred = project_utils.get_popular_element(pred)
            major_preds.append(major_label)
            pred_prob.append(num_models_pred / num_models)
        ensemble_pred_time = time.time() - start_time
        pred_time.update({"Ensemble": ensemble_pred_time})
        print("{}:: Model Ensemble predict done. Time : {:.2f} seconds".format(
            self.class_name, ensemble_pred_time))

        pred_class_id_df["Ensemble"] = major_preds
        pred_label_df = pred_class_id_df.applymap(lambda class_id: self.class_mid_map.get(class_id))
        pred_class_id_df["Predict Probability"] = pred_prob
        pred_label_df["Predict Probability"] = pred_prob

        exec_time = time.time() - start_time
        print("{}:: {} models predict done. Time {:.2f} seconds".format(
            self.class_name, len(self.models), exec_time))

        return pred_class_id_df, pred_label_df, pred_time

    def evaluate_from_image_paths(self, test_image_paths, labels=None):
        face_encodings = []
        available_labels = []
        for i, img_path in enumerate(test_image_paths):
            fencoding = get_face_encodings(image_path=img_path)
            if len(fencoding) > 0:
                face_encodings.append(fencoding)
                if labels is None:
                    available_labels.append(utils.get_parent_name(img_path))
                else:
                    available_labels.append(labels[i])

        print("available_labels: ", available_labels)
        X_test = np.array(face_encodings)
        y_test = np.array([self.mid_class_map.get(mid) for mid in available_labels])

        pred_label_df = self.evaluate(X_test, y_test)
        pred_label_df["True Label"] = available_labels

        print("{}:: Predict result ".format(self.class_name))
        print(pred_label_df.head())

    def evaluate(self, X_test, y_test, metrics={}):
        start_time = time.time()
        print("{}:: Start evaluate on {} samples with metrics : {}".format(
            self.class_name, X_test.shape[0], list(metrics.keys())))

        pred_class_id_df, pred_label_df, pred_times = self.predict(X_test)
        y_pred = pred_class_id_df.iloc[:, 0].values

        print("{}:: Predict result".format(self.class_name))
        print(pred_label_df.head())

        accuracy = accuracy_score(y_test, y_pred)
        print("{}:: Accuracy : {:.4f} %".format(self.class_name, accuracy * 100))

        exec_time = time.time() - start_time
        print("{}:: Evaluate {} models done. Time {:.2f} seconds".format(
            self.class_name, len(self.models), exec_time))

        return pred_label_df

    def add_model(self, name, model):
        self.models.update({name: model})
        print("{}:: Add model {} done".format(self.class_name, name))

    def remove_model(self, name):
        if name in self.models:
            del self.models[name]
            print("{}:: Remove model {} done".format(self.class_name, name))
        else:
            print("{}:: Can not remove model {} because it is not in current models".format(
                self.class_name, name))

    def save_model(self):
        pass

    def load_model(self):
        pass


def test_pipeline_model():
    training_data_dir = "../Temp/Dataset/Version2"
    face_encoding_dir = "../Temp/Dataset/Process/face_encodings"
    mid_name_path = "../Temp/Dataset/Process/MID_Name.json"

    model = BaseLine1Model(
        training_data_dir=training_data_dir,
        face_encoding_dir=face_encoding_dir,
        mid_name_path=mid_name_path
    )

    # Add KNN model
    knn_model = KNeighborsClassifier()
    model.add_model("KNN", knn_model)

    model.train()

    # Test
    test_image_dir = "../Temp/Dataset/Test/DV - Quốc Quân"
    test_image_paths = utils.get_file_paths(test_image_dir)
    model.evaluate_from_image_paths(test_image_paths=test_image_paths)


if __name__ == "__main__":
    pass
    # dataset_dir = "../Temp/Dataset/Original"
    # save_dir = "../Temp/Dataset/Process"

    # ap = argparse.ArgumentParser()
    # ap.add_argument("--dataset_dir", required=True, help="Directory path of dataset contain multi folder that each folder represent unique person")
    # ap.add_argument("--save_dir", required=True, help="Directory path to save face encodings")
    #
    # args = vars(ap.parse_args())
    # dataset_dir = args["dataset_dir"]
    # save_dir = args["save_dir"]
    #
    # save_face_encoding(dataset_dir=dataset_dir, save_dir=save_dir)

    # face_encoding_dir = "../Dataset/Process/face_encodings"
    # src_dataset_dir = "/home/quanchu/Dataset/FaceImagCroppedWithAlignmentShorten"
    # dst_dataset_dir = "/home/quanchu/Dataset/Version2"
    #
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--face_encoding_dir", required=True, help="Directory path contain face encodings")
    # ap.add_argument("--src_dataset_dir", required=True, help="Directory path contain source dataset")
    # ap.add_argument("--dst_dataset_dir", required=True, help="Directory path contain destination dataset")
    #
    # args = vars(ap.parse_args())
    # face_encoding_dir = args["face_encoding_dir"]
    # src_dataset_dir = args["src_dataset_dir"]
    # dst_dataset_dir = args["dst_dataset_dir"]

    # create_subset_data(
    #     face_encoding_dir=face_encoding_dir,
    #     src_dataset_dir=src_dataset_dir,
    #     dst_dataset_dir=dst_dataset_dir
    # )

    test_pipeline_model()

