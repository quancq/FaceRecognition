import keras
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
import os
import time
import pandas as pd
from utils_dir import utils
import argparse


class MyResNet:
    def __init__(self, image_size, num_epochs, batch_size, dataset_dir, save_dir):
        self.image_size = image_size
        self.input_shape = (self.image_size, self.image_size, 3)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = os.path.join(save_dir, utils.get_time_str())
        self.dataset_dir = dataset_dir
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
            vertical_flip=True,
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
        resnet_base = ResNet50(include_top=False, input_shape=self.input_shape)

        # Freeze low layer
        for layer in resnet_base.layers[:-10]:
            layer.trainable = False

        # Show trainable status of each layers
        print("\nAll layers of resnet50 base")
        for layer in resnet_base.layers:
            print("Layer : {} - Trainable : {}".format(layer, layer.trainable))

        self.num_classes = len(train_generator.class_indices)
        model = Sequential()
        model.add(resnet_base)
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        print("\nFinal model summary")
        model.summary()

        # Compile model
        model.compile(
            loss="categorical_crossentropy",
            metrics=["acc"],
            optimizer=Adam(lr=1e-4)
        )

        classes = [_ for _ in range(self.num_classes)]
        for c in train_generator.class_indices:
            classes[train_generator.class_indices[c]] = c

        model.classes = classes

        # Define callbacks
        save_model_dir = os.path.join(self.save_dir, "Model")
        loss_path = os.path.join(save_model_dir, "{epoch:02d}-{val_loss:.2f}.h5")
        loss_checkpoint = ModelCheckpoint(
            filepath=loss_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True
        )

        acc_path = os.path.join(save_model_dir, "{epoch:02d}-{val_acc:.2f}.h5")
        acc_checkpoint = ModelCheckpoint(
            filepath=acc_path,
            monitor="val_acc",
            verbose=1,
            save_best_only=True
        )
        callbacks = [loss_checkpoint, acc_checkpoint]

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
        data = [acc, val_acc, loss, val_loss]
        columns = ["Accuracy", "Valid_Accuracy", "Loss", "Valid_Loss"]
        df = pd.DataFrame(data, columns=columns)
        save_path = os.path.join(self.save_dir, "history.csv")
        utils.save_csv(df, save_path)

        exec_time = time.time() - start_time
        print("\nTrain model done. Time : {:.2f} seconds".format(exec_time))


def train():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--save_dir", default="./Experiment")
    ap.add_argument("--num_epochs", default=100)
    ap.add_argument("--image_size", default=160)
    ap.add_argument("--batch_size", default=128)

    args = vars(ap.parse_args())
    dataset_dir = args["dataset_dir"]
    save_dir = args["save_dir"]
    num_epochs = int(args["num_epochs"])
    image_size = int(args["image_size"])
    batch_size = int(args["batch_size"])

    model = MyResNet(
        image_size=image_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        save_dir=save_dir
    )
    model.train()


if __name__ == "__main__":
    train()

