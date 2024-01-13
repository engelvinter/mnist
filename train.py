import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import argparse
from dataclasses import dataclass
from rich.console import Console

import numpy
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

from functions import adjust_images, normalize_images

class DataSet:
    train_images : numpy.ndarray
    train_labels : numpy.ndarray
    test_images : numpy.ndarray
    test_labels : numpy.ndarray

console = Console()

def load_data() -> DataSet:
    data = DataSet()
    console.log("Loading mnist data")
    (data.train_images, data.train_labels), (data.test_images, data.test_labels) = mnist.load_data()
    return data

def create_one_layer_model() -> Sequential:
    console.log("Creating model")
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten the input images
    model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 units (for each digit) and softmax activation

    model.compile(optimizer='adam',  # You can use other optimizers like 'sgd' or 'rmsprop'
              loss='categorical_crossentropy',  # For multi-class classification problems
              metrics=['accuracy'])
    
    return model

def create_two_layer_model() -> Sequential:
    console.log("Creating model")
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten the input images
    model.add(Dense(256, activation='relu'))  # Dense layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 units (for each digit) and softmax activation

    model.compile(optimizer='adam',  # You can use other optimizers like 'sgd' or 'rmsprop'
              loss='categorical_crossentropy',  # For multi-class classification problems
              metrics=['accuracy'])
    
    return model

def train(model : Sequential, data_set : DataSet, nbr_epochs : int) -> Sequential:
    with console.status("Adjusting mnist digits") as status:
        train_images = adjust_images(data_set.train_images)
        test_images = adjust_images(data_set.test_images)

    console.log("Normalizing images")
    train_images = normalize_images(data_set.train_images)
    test_images = normalize_images(data_set.test_images)

    console.log("Hot encoding output labels")
    hot_train_labels = to_categorical(data_set.train_labels)
    hot_test_labels = to_categorical(data_set.test_labels)

    console.log("Start training:")
    model.fit(train_images, hot_train_labels, epochs=nbr_epochs, batch_size=64, validation_split=0.1)

    console.log("Evaluate:")
    test_loss, test_accuracy = model.evaluate(test_images, hot_test_labels)
    console.log(f'Test Accuracy is {test_accuracy * 100:.2f}%')


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="The console programs trains a given neural network model using the mnist training set")
    parser.add_argument("model_result_file", help="the produced model including weights after training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs during training")
    args = parser.parse_args()

    data_set = load_data()
    model = create_two_layer_model()

    train(model, data_set, args.epochs)
    model.save(args.model_result_file)

if __name__ == "__main__":
    main()
