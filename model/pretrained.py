import keras
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 12})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, BatchNormalization, Input, ReLU
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
import os
import time
import pandas as pd
from utils_dir import utils
import argparse
from utils_dir.image_loader import generate_batch
from model.siamese_network import contrastive_loss, get_siamese_model, accuracy


class MyResNet:
    def __init__(self, image_size, num_epochs, batch_size, dataset_dir, save_dir,
                 model_name="VGG16", num_trainable_layer=5, lr=1e-3,
                 optimizer="Adam", model_path=None, is_siamese=True):
        self.model_name = model_name
        self.image_size = image_size
        self.input_shape = (self.image_size, self.image_size, 3)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = os.path.join(save_dir, utils.get_time_str())
        self.dataset_dir = dataset_dir
        self.optimizer = optimizer
        self.num_trainable_layer = num_trainable_layer
        self.lr = lr
        self.model_path = model_path
        self.is_siamese = is_siamese

        self.train_dir = os.path.join(dataset_dir, "Train")
        self.valid_dir = os.path.join(dataset_dir, "Valid")
        self.test_dir = os.path.join(dataset_dir, "Test")

    def train(self):
        start_time = time.time()

        # Setup generator
        if self.is_siamese:
            train_generator = generate_batch(
                dataset_dir=self.train_dir,
                batch_size=self.batch_size,
                image_size=self.image_size
            )
            valid_generator = generate_batch(
                dataset_dir=self.valid_dir,
                batch_size=self.batch_size,
                image_size=self.image_size
            )

            self.num_classes = len(utils.get_dir_names(self.train_dir))

        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
            )

            valid_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                directory=self.train_dir,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
            )

            valid_generator = valid_datagen.flow_from_directory(
                directory=self.valid_dir,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
            )
            self.num_classes = len(train_generator.class_indices)

        optimizer = Adam
        if self.optimizer == "Adam":
            optimizer = Adam
        elif self.optimizer == "RMSProp":
            optimizer = RMSprop
        model = None
        # Check training from scratch or continue training
        if self.model_path is not None:
            model = load_model(self.model_path)
        else:
            if self.model_name == "VGG16":
                model_base = VGG16(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "ResNet50":
                model_base = ResNet50(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "DenseNet121":
                model_base = DenseNet121(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "InceptionV3":
                model_base = InceptionV3(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "InceptionResNetV2":
                model_base = InceptionResNetV2(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "Xception":
                model_base = Xception(include_top=False, input_shape=self.input_shape)
            elif self.model_name == "Scratch":
                model_base = Sequential()
                model_base.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=self.input_shape))
                model_base.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
                model_base.add(MaxPool2D())
                model_base.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
                model_base.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
                model_base.add(MaxPool2D())
                model_base.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
                model_base.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
                model_base.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
                model_base.add(MaxPool2D())
                model_base.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
                model_base.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
                model_base.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
                model_base.add(MaxPool2D())
                self.num_trainable_layer = len(model_base.layers)
            else:
                print("Model name {} is not valid ".format(self.model_name))
                return 0

            # Freeze low layer
            for layer in model_base.layers[:-self.num_trainable_layer]:
                layer.trainable = False

            # Show trainable status of each layers
            print("\nAll layers of {} ".format(self.model_name))
            for layer in model_base.layers:
                print("Layer : {} - Trainable : {}".format(layer, layer.trainable))

            model = Sequential()
            model.add(model_base)
            model.add(Flatten())
            # model.add(Dense(50, activation="relu"))
            # model.add(Dropout(0.25))
            model.add(Dense(self.num_classes, activation="softmax"))

            # Compile model
            model.compile(
                loss="categorical_crossentropy",
                metrics=["acc"],
                optimizer=optimizer(lr=self.lr)
            )

        if self.is_siamese:
            model = get_siamese_model(model)
            model.compile(
                loss=contrastive_loss,
                metrics=[accuracy],
                optimizer=optimizer(lr=self.lr)
            )

        print("\nFinal model summary")
        model.summary()

        # classes = [_ for _ in range(self.num_classes)]
        # for c in train_generator.class_indices:
        #     classes[train_generator.class_indices[c]] = c
        #
        # model.classes = classes

        # Define callbacks
        save_model_dir = os.path.join(self.save_dir, "Model_{}".format(self.model_name))
        utils.make_dirs(save_model_dir)
        # loss_path = os.path.join(save_model_dir, "epochs_{epoch:02d}-val_loss_{val_loss:.2f}.h5")
        # loss_checkpoint = ModelCheckpoint(
        #     filepath=loss_path,
        #     monitor="val_loss",
        #     verbose=1,
        #     save_best_only=True
        # )

        acc_path = os.path.join(save_model_dir, "epochs_{epoch:02d}-val_acc_{val_acc:.2f}.h5")
        acc_checkpoint = ModelCheckpoint(
            filepath=acc_path,
            monitor="val_acc",
            verbose=1,
            save_best_only=True
        )
        callbacks = [acc_checkpoint]

        # Train model
        print("Start train model from {} ...".format(
            "{} pretrained".format(self.model_name) if self.model_path is None else self.model_path))
        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.samples/train_generator.batch_size,
            epochs=self.num_epochs,
            validation_data=valid_generator,
            validation_steps=valid_generator.samples/valid_generator.batch_size,
            callbacks=callbacks
        )

        # Save model
        save_path = os.path.join(save_model_dir, "final_model.h5")
        model.save(save_path)

        # Save history
        acc, val_acc = history.history["acc"], history.history["val_acc"]
        loss, val_loss = history.history["loss"], history.history["val_loss"]
        train_stats = dict(Loss=loss, Valid_Loss=val_loss, Accuracy=acc, Valid_Accuracy=val_acc)
        df = pd.DataFrame(train_stats)
        save_path = os.path.join(self.save_dir, "History.csv")
        utils.save_csv(df, save_path)

        exec_time = time.time() - start_time
        print("\nTrain model {} done. Time : {:.2f} seconds".format("{} pretrained".format(self.model_name) if self.model_path is None else self.model_path, exec_time))


def train():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="VGG16")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--save_dir", default="./Experiment")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--num_epochs", default=100)
    ap.add_argument("--image_size", default=160)
    ap.add_argument("--batch_size", default=128)
    ap.add_argument("--num_trainable_layer", default=5)
    ap.add_argument("--lr", default=0.001)
    ap.add_argument("--opt", default="Adam")
    ap.add_argument("--is_siamese", default="y")

    args = vars(ap.parse_args())
    model_name = args["model_name"]
    dataset_dir = args["dataset_dir"]
    save_dir = args["save_dir"]
    model_path = args["model_path"]
    num_epochs = int(args["num_epochs"])
    image_size = int(args["image_size"])
    batch_size = int(args["batch_size"])
    num_trainable_layer = int(args["num_trainable_layer"])
    lr = float(args["lr"])
    opt = args["opt"]
    is_siamese = args["is_siamese"]

    is_siamese = True if is_siamese == "y" else False

    model = MyResNet(
        image_size=image_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        save_dir=save_dir,
        model_name=model_name,
        num_trainable_layer=num_trainable_layer,
        lr=lr,
        optimizer=opt,
        model_path=model_path,
        is_siamese=is_siamese
    )
    model.train()


if __name__ == "__main__":
    train()


