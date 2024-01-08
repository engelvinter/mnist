import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import argparse
from rich.console import Console

from functions import read_image_files, adjust_images, normalize_images

from keras import models
import numpy as np

console = Console()

def main():
    parser = argparse.ArgumentParser(description="The console programs predicts a result using the given input data and given model")
    parser.add_argument("model_input_file", help="the model to use in prediction")
    parser.add_argument("path_digits", help="path to digits to use as input data")
    args = parser.parse_args()

    console.log("Loading model with weights")
    model = models.load_model(args.model_input_file)
    console.log("Reading digit files")
    filenames, images = read_image_files(args.path_digits)
    console.log("Adjusting digits")
    adjusted_images = adjust_images(images)
    console.log("Normalizing images")
    normalized_images = normalize_images(adjusted_images)
    console.log("Prediciting:")
    prediction = model.predict(normalized_images)
    result = np.argmax(prediction, axis = 1)
    console.log(result)
    console.log(filenames)

if __name__ == "__main__":
    main()