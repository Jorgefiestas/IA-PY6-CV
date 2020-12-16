from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

EPOCHS = 20

def load_data(base_path):
    data = []
    labels = []
    for dir_path in os.listdir(base_path):
        print(dir_path)
        img_paths = list(paths.list_images( base_path + os.path.sep + dir_path))
        for img_path in img_paths:
            label = img_path.split(os.path.sep)[-2]
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)

            data.append(img)
            labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def train_model(data, labels, EPOCHS = 1, BS = 32, LR=1e-4):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
            test_size=0.20, stratify=labels, random_state=42)

    aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

    std_model = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

    head_model = std_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    model = Model(inputs=std_model.input, outputs=head_model)
    for layer in std_model.layers:
            layer.trainable = False


    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=["accuracy"])

    fitted_model = model.fit(
            aug.flow(X_train, Y_train, batch_size=BS),
            steps_per_epoch=len(X_train) // BS,
            validation_data=(X_test, Y_test),
            validation_steps=len(X_test) // BS,
            epochs=EPOCHS)

    f_idx = model.predict(X_test, batch_size=BS)
    f_idx = np.argmax(f_idx, axis = 1)
    model.save("../models/model_cnn", save_format="h5")
    return fitted_model

def plot_curve(model, N):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), model.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Learning curve")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss vs Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("../images/learning_curve.png")


data, labels = load_data("../dataset/data/")
fitted_model = train_model(data = data, labels = labels, EPOCHS = EPOCHS)
plot_curve(fitted_model, N = EPOCHS)
