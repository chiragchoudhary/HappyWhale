import tensorflow as tf


def process_files(files, labels, shuffle):
    """Returns a single image tensor."""

    height = 64
    width = 64
    num_channels = 3

    filename, label = tf.train.slice_input_producer([files, labels], shuffle=shuffle)

    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize_images(image, [height, width])
    return image, label


def get_batches(images, labels, batch_size=32, shuffle=True):
    """Returns a batch of images and their corresponding labels."""

    image, label = process_files(images, labels, shuffle)
    img_batch, labels_batch = tf.train.batch(
                                [image, label],
                                batch_size,
                                capacity=4000,
                                )

    return img_batch, labels_batch
