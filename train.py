import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import argparse
from dataclasses import dataclass
from rich.console import Console
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

from functions import adjust_images, normalize_images, is_running_in_jupyter

matplotlib.use('agg')  # Set the backend to 'agg'

class DataSet:
    train_images : np.ndarray
    train_labels : np.ndarray
    test_images : np.ndarray
    test_labels : np.ndarray

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

def prepare(data_set : DataSet) -> DataSet:
    prep_set = DataSet()
    with console.status("Adjusting mnist digits") as status:
        prep_set.train_images = adjust_images(data_set.train_images)
        prep_set.test_images = adjust_images(data_set.test_images)

    console.log("Normalizing images")
    prep_set.train_images = normalize_images(prep_set.train_images)
    prep_set.test_images = normalize_images(prep_set.test_images)

    console.log("Hot encoding output labels")
    prep_set.train_labels = to_categorical(data_set.train_labels)
    prep_set.test_labels = to_categorical(data_set.test_labels)

    return prep_set

def train(model : Sequential, data_set : DataSet, nbr_epochs : int) -> Sequential:
    console.log("Start training:")
    model.fit(data_set.train_images, data_set.train_labels, epochs=nbr_epochs, batch_size=64, validation_split=0.1)

    return model

def evaluate(model : Sequential, data_set : DataSet):
    console.log("Evaluate:")
    test_loss, test_accuracy = model.evaluate(data_set.test_images, data_set.test_labels)
    console.log(f'Test Accuracy is {test_accuracy * 100:.2f}%')
    predict_labels = model.predict(data_set.test_images)

    # Convert from hot encoded to classification
    test_classes = np.argmax(data_set.test_labels, axis = 1)
    predict_classes = np.argmax(predict_labels, axis = 1)

    console.log("Create confusion matrix")

    # Create the confusion matrix
    conf_matrix = confusion_matrix(test_classes, predict_classes)

    # Visualize the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if is_running_in_jupyter():
        plt.show()
    else:
        plt.savefig('confusion_matrix.png')

    console.log('\nClassification Report:')
    console.log(classification_report(test_classes, predict_classes))

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="The console programs trains a given neural network model using the mnist training set")
    parser.add_argument("model_result_file", help="the produced model including weights after training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs during training")
    args = parser.parse_args()

    data_set = load_data()
    model = create_two_layer_model()

    prepared_data_set = prepare(data_set)
    train(model, prepared_data_set, args.epochs)
    model.save(args.model_result_file)
    evaluate(model, prepared_data_set)

if __name__ == "__main__":
    main()
