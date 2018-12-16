import keras
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 12})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
import os
import time
import pandas as pd
from utils_dir import utils
import argparse


class MyResNet:
    def __init__(self, image_size, num_epochs, batch_size, dataset_dir, save_dir,
                 model_name="VGG16", num_trainable_layer=5, lr=1e-3, optimizer="Adam"):
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

        self.train_dir = os.path.join(dataset_dir, "Train")
        self.valid_dir = os.path.join(dataset_dir, "Valid")
        self.test_dir = os.path.join(dataset_dir, "Test")

    def train(self):
        start_time = time.time()

        # Setup generator
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
        if self.model_name == "VGG16":
            model_base = VGG16(include_top=False, input_shape=self.input_shape)
        elif self.model_name == "ResNet50":
            model_base = ResNet50(include_top=False, input_shape=self.input_shape)
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

        self.num_classes = len(train_generator.class_indices)
        model = Sequential()
        model.add(model_base)
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation="softmax"))

        print("\nFinal model summary")
        model.summary()

        # Compile model
        optimizer = Adam
        if self.optimizer == "Adam":
            optimizer = Adam
        elif self.optimizer == "RMSProp":
            optimizer = RMSprop

        model.compile(
            loss="categorical_crossentropy",
            metrics=["acc"],
            optimizer=optimizer(lr=self.lr)
        )

        classes = [_ for _ in range(self.num_classes)]
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        model.classes = classes

        # Define callbacks
        save_model_dir = os.path.join(self.save_dir, "Model")
        utils.make_dirs(save_model_dir)
        loss_path = os.path.join(save_model_dir, "epochs_{epoch:02d}-val_loss_{val_loss:.2f}.h5")
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
        print("Start train model ...")
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
        train_stats = [acc, loss]
        columns = ["Accuracy", "Loss"]
        df = pd.DataFrame(train_stats, columns=columns)
        save_path = os.path.join(self.save_dir, "History_Train.csv")
        utils.save_csv(df, save_path)

        val_stats = [val_acc, val_loss]
        columns = ["Valid_Accuracy", "Valid_Loss"]
        df = pd.DataFrame(val_stats, columns=columns)
        save_path = os.path.join(self.save_dir, "History_Valid.csv")
        utils.save_csv(df, save_path)

        exec_time = time.time() - start_time
        print("\nTrain model done. Time : {:.2f} seconds".format(exec_time))


def train():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="VGG16")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--save_dir", default="./Experiment")
    ap.add_argument("--num_epochs", default=100)
    ap.add_argument("--image_size", default=160)
    ap.add_argument("--batch_size", default=128)
    ap.add_argument("--num_trainable_layer", default=5)
    ap.add_argument("--lr", default=0.001)
    ap.add_argument("--opt", default="Adam")

    args = vars(ap.parse_args())
    model_name = args["model_name"]
    dataset_dir = args["dataset_dir"]
    save_dir = args["save_dir"]
    num_epochs = int(args["num_epochs"])
    image_size = int(args["image_size"])
    batch_size = int(args["batch_size"])
    num_trainable_layer = int(args["num_trainable_layer"])
    lr = float(args["lr"])
    opt = args["opt"]

    model = MyResNet(
        image_size=image_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        save_dir=save_dir,
        model_name=model_name,
        num_trainable_layer=num_trainable_layer,
        lr=lr,
        optimizer=opt
    )
    model.train()


if __name__ == "__main__":
    train()


