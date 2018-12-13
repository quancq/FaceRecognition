import numpy as np
import argparse
from utils_dir import utils
from settings import RANDOM_STATE
from model.baseline1 import BaseLine1Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def evaluate_baseline1(training_data_dir, test_data_dir, face_encoding_dir,
                    mid_name_path, experiment_dir, model_dir, model_names=[]):
    np.random.seed(RANDOM_STATE)
    # training_data_dir = "./Temp/Dataset/Version2"
    # face_encoding_dir = "./Temp/Dataset/Process/face_encodings"
    # mid_name_path = "./Temp/Dataset/Process/MID_Name.json"
    # experiment_dir = "./Temp/Experiment"

    model = BaseLine1Model(
        training_data_dir=training_data_dir,
        face_encoding_dir=face_encoding_dir,
        mid_name_path=mid_name_path,
        experiment_dir=experiment_dir,
        mode="test"
    )
    model.load_model(model_dir, model_names)

    # Test
    dirs = utils.get_dir_paths(test_data_dir)
    test_image_paths = []
    for dir in dirs:
        test_image_paths.extend(utils.get_file_paths(dir))

    # Evaluate model after load model from disk
    model.evaluate_from_image_paths(test_image_paths=test_image_paths, save_result=True)


if __name__ == "__main__":
    # training_data_dir = "./Dataset/Train_Test1/Train"
    # test_data_dir = "./Dataset/Train_Test1/Test"
    # face_encoding_dir = "./Dataset/Process/face_encodings"
    # mid_name_path = "./Dataset/Process/MID_Name.json"
    # experiment_dir = "./Experiment"

    ap = argparse.ArgumentParser()
    ap.add_argument("--training_data_dir", required=True)
    ap.add_argument("--test_data_dir", required=True)
    ap.add_argument("--face_encoding_dir", required=True)
    ap.add_argument("--mid_name_path", required=True, help="Path of file contain mapping from mid to name")
    ap.add_argument("--experiment_dir", help="Directory to save results", default="./Experiment")
    ap.add_argument("--model_dir", required=True, help="Directory contain models")
    ap.add_argument("--model_names", help="List models to train", default="KNN")

    args = vars(ap.parse_args())
    training_data_dir = args["training_data_dir"]
    test_data_dir = args["test_data_dir"]
    face_encoding_dir = args["face_encoding_dir"]
    mid_name_path = args["mid_name_path"]
    experiment_dir = args["experiment_dir"]
    model_dir = args["model_dir"]
    model_names = args["model_names"]

    model_names = model_names.split("-")
    evaluate_baseline1(
        training_data_dir=training_data_dir,
        test_data_dir=test_data_dir,
        face_encoding_dir=face_encoding_dir,
        mid_name_path=mid_name_path,
        experiment_dir=experiment_dir,
        model_dir=model_dir,
        model_names=model_names
    )
