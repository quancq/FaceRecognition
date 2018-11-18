import numpy as np
import argparse
from utils import utils
from settings import RANDOM_STATE
from model.baseline1 import BaseLine1Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def train_baseline1(training_data_dir, test_data_dir,
                    face_encoding_dir, mid_name_path, experiment_dir):
    np.random.seed(RANDOM_STATE)
    # training_data_dir = "./Temp/Dataset/Version2"
    # face_encoding_dir = "./Temp/Dataset/Process/face_encodings"
    # mid_name_path = "./Temp/Dataset/Process/MID_Name.json"
    # experiment_dir = "./Temp/Experiment"

    model = BaseLine1Model(
        training_data_dir=training_data_dir,
        face_encoding_dir=face_encoding_dir,
        mid_name_path=mid_name_path,
        experiment_dir=experiment_dir
    )

    # Add KNN Grid search
    knn_gs = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid={
            "n_neighbors": np.arange(10, 30, 2)
        },
        n_jobs=-1,
        cv=3,
        scoring="accuracy"
    )
    model.add_model("KNN_GS", knn_gs)

    # Add Logistic model
    # lr_model = LogisticRegression(
    #     C=0.6,
    #     solver="lbfgs",
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1
    # )
    # model.add_model("Logistic", lr_model)

    # Add KNN model
    knn_model = KNeighborsClassifier(n_neighbors=40, n_jobs=-1)
    model.add_model("KNN", knn_model)

    # Add Linear SVM model
    linear_svm_model = LinearSVC(random_state=RANDOM_STATE)
    model.add_model("LinearSVM", linear_svm_model)

    # Add Kernel SVM model
    # kernel_svm_model = SVC(
    #     C=0.001,
    #     gamma=0.1,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("KernelSVM", kernel_svm_model)

    # Add Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=50,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.add_model("RandomForest", rf_model)

    # Add Extra Tree model
    et_model = ExtraTreesClassifier(
        n_estimators=30,
        max_depth=50,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.add_model("ExtraTree", et_model)

    # Add Fully connected model
    fc_model = MLPClassifier(
        hidden_layer_sizes=(256,),
        batch_size=64
    )
    model.add_model("FC", fc_model)

    model.train()

    # Test
    dirs = utils.get_dir_paths(test_data_dir)
    test_image_paths = []
    for dir in dirs:
        test_image_paths.extend(utils.get_file_paths(dir))
    # model.evaluate_from_image_paths(test_image_paths=test_image_paths)

    # Save model
    save_dir = model.save_model()

    # # Load model
    model.load_model(model_dir=save_dir)

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

    args = vars(ap.parse_args())
    training_data_dir = args["training_data_dir"]
    test_data_dir = args["test_data_dir"]
    face_encoding_dir = args["face_encoding_dir"]
    mid_name_path = args["mid_name_path"]
    experiment_dir = args["experiment_dir"]

    train_baseline1(
        training_data_dir=training_data_dir,
        test_data_dir=test_data_dir,
        face_encoding_dir=face_encoding_dir,
        mid_name_path=mid_name_path,
        experiment_dir=experiment_dir
    )
