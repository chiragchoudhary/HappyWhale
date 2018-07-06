import tensorflow as tf
import os
import numpy as np
import fnmatch
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def get_train_val_split(labels_file, split_ratio=0.2):
    """Returns training and validation sets of images and their respective labels."""

    labels = pd.read_csv(labels_file, delimiter=',')
    print "Number of unique whale species >= {}".format(labels['Id'].nunique())

    label_encoder = LabelBinarizer()
    one_hot_labels = label_encoder.fit_transform(labels['Id'].tolist())

    decoded_labels = np.argmax(one_hot_labels, axis=1)
    mapping = {}
    for i, row in labels['Id'].iteritems():
        mapping[decoded_labels[i]] = row
    with open(os.path.join(os.getcwd(), "data", "label_mapping.pkl"), "w+") as f:
        pkl.dump(mapping, f)

    train_x, val_x, train_y, val_y = train_test_split(labels['Image'], one_hot_labels, test_size=split_ratio)
    return train_x, val_x, train_y, val_y


def process_files(files, labels, shuffle, num_epochs=None):
    """Returns a single image tensor."""

    height = 512
    width = 512
    num_channels = 3

    filename, label = tf.train.slice_input_producer([files, labels], shuffle=shuffle, num_epochs=num_epochs)
    prefix = os.path.join(os.getcwd(), "data", "img", "")
    filename = tf.string_join([prefix, filename])
    image = tf.read_file(filename)

    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize_images(image, [height, width])
    return image, label


def get_labeled_batches(images, labels, batch_size=32, shuffle=True, num_epochs=None):
    """Returns a batch of images and their corresponding labels."""

    image, label = process_files(images, labels, shuffle=shuffle, num_epochs=num_epochs)
    img_batch, labels_batch = tf.train.batch(
                                [image, label],
                                batch_size,
                                capacity=4000,
                                allow_smaller_final_batch=True,
                                num_threads=12
                                )

    return img_batch, labels_batch


def get_jpeg_files(img_dir):
    """Returns a list of all .jpg files in the directory."""

    pattern = "*.jpg"
    images = fnmatch.filter(os.listdir(img_dir), pattern)
    return images


def get_test_batches(images, batch_size=32, num_epochs=None):
    """Returns a batch of images from test set."""

    height = 512
    width = 512
    num_channels = 3

    filename = tf.train.slice_input_producer([images], shuffle=False, num_epochs=num_epochs)[0]
    prefix = os.path.join(os.getcwd(), "data", "test", "")
    filename = tf.string_join([prefix, filename])
    image = tf.read_file(filename)

    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize_images(image, [height, width])
    img_batch = tf.train.batch(
        [image],
        batch_size,
        capacity=4000,
        allow_smaller_final_batch=True,
        num_threads=4,
    )

    return img_batch


def get_top_k_classes(top_k_predictions):
    """Returns the list of top k labels for each image."""

    with open(os.path.join(os.getcwd(), "data", "label_mapping.pkl"), "r") as f:
        mapping = pkl.load(f)

    top_k_labels = []
    for batch_prediction in top_k_predictions:
        for prediction in batch_prediction:
            labels = []
            for index in prediction:
                labels.append(mapping[index])
            labels = ' '.join(labels)
            top_k_labels.append(labels)

    return top_k_labels
