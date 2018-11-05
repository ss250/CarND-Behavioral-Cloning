import csv
import cv2
import numpy as np
import argparse
from models import ModelLoader

from keras.callbacks import EarlyStopping, ModelCheckpoint


def load_images(csv_path, img_path, expected_shape):

    lines = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append(line)


    images = []
    measurements = []

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]

        current_path = img_path + filename

        image = np.array(cv2.imread(current_path))

        if image.shape != expected_shape:
            continue

        # print("reading image: " + current_path)

        images.append(image)
        # print(image)

        measurement = float(line[3])
        # print("adding measurements: " + str(measurements))
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    print(X_train.shape, y_train.shape)

    return X_train, y_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of network to use.'
    )
    args = parser.parse_args()

    CSV_PATH = "/home/alastair/git_projects/udacity/CarND-Behavioral-Cloning-P3/data/driving_log.csv"
    IMG_PATH = "/home/alastair/git_projects/udacity/CarND-Behavioral-Cloning-P3/data/IMG/"
    img_shape = (160, 320, 3)

    X_train, y_train = load_images(CSV_PATH, IMG_PATH, img_shape)

    
    early_stopper = EarlyStopping(patience=2)
    checkpointer = ModelCheckpoint(
        filepath="checkpoints/" + args.model_name + '-{epoch:03d}-{loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    loader = ModelLoader(args.model_name, (160, 320, 3))

    model = loader.model

    model.fit(X_train, y_train, callbacks=[early_stopper, checkpointer], epochs=50, validation_split=0.2, shuffle=True)

    model.save(args.model_name + ".h5")