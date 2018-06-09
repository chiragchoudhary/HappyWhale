import tensorflow as tf


def get_batches(images, labels, batch_size=32):
    """Returns a batch of images and their corresponding labels."""

    img_name_batch, labels_batch = tf.train.shuffle_batch(
                                [images, labels],
                                batch_size,
                                capacity=50000,
                                min_after_dequeue=10000
                                )
    images_batch = []
    for img_file in images:
        image = tf.image.decode_jpeg(img_file)

        images_batch.append(image)

    images_batch = tf.image.resize_images(images_batch, [32, 32])
    images_batch = tf.convert_to_tensor(images_batch, dtype=tf.float64)

    return images_batch, labels_batch
