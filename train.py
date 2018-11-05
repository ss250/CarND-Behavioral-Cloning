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
        center_image = np.array(cv2.imread(img_path +  line[0].split('/')[-1]))
#         left_image   = np.array(cv2.imread(img_path +  line[1].split('/')[-1]))
#         right_image  = np.array(cv2.imread(img_path +  line[2].split('/')[-1]))

#         if (center_image.shape != expected_shape) or (left_image.shape != expected_shape) or (right_image.shape != expected_shape):
#             continue
        if center_image.shape != expected_shape:
            continue

        images.append(center_image)
        measurement = float(line[3])
        print(measurement)
        measurements.append(measurement)

    aug_images, aug_measurements = [], []
    for image, measurement in zip(images, measurements):
        aug_images.append(image)
        aug_measurements.append(measurement)
        
        aug_images.append(cv2.flip(image, 1))
        aug_measurements.append(measurement*-1.0)
        
    X_train = np.array(aug_images)
    y_train = np.array(aug_measurements)

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

    CSV_PATH = "./data/driving_log.csv"
    IMG_PATH = "./data/IMG/"
    img_shape = (160, 320, 3)

    X_train, y_train = load_images(CSV_PATH, IMG_PATH, img_shape)

    
    early_stopper = EarlyStopping(patience=2)
    checkpointer = ModelCheckpoint(
        filepath="checkpoints/" + args.model_name + '.hdf5',
        verbose=1,
        save_best_only=True)

    loader = ModelLoader(args.model_name, (160, 320, 3))

    model = loader.model

    model.fit(X_train, y_train, callbacks=[early_stopper, checkpointer], epochs=50, validation_split=0.2, shuffle=True)

    model.save(args.model_name + ".h5")