import os

import tensorflow as tf
from processing import get_batches


class CNN:
    """Defines a Convolution Neural Network with 1 convolution layer, and 1 fully connected layer"""

    def __init__(self, batch_size=16):
        """Define the CNN model architecture."""

        self.model_name = "CNN"

        self.batch_size = batch_size
        self.images = tf.placeholder(tf.float64, shape=[None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float64, shape=[None, 1])
        self.out = tf.layers.conv3d(self.images, 16, [3, 3], [2, 2], padding="same")
        self.out = tf.nn.relu(self.out)
        self.out = tf.layers.max_pooling3d(self.out, 2, 2)
        self.out = tf.reshape(self.out, [-1, 32 * 32 * 16])
        self.logits = tf.layers.dense(self.out, 4251)

    def init(self):
        """Initialize the model variables."""
        # TODO
        return

    def train(self, images, labels, epochs=10):
        """Trains the model on training data."""

        num_batches = int(images.get_shape()[0]/self.batch_size)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        optimizer = tf.train.AdamOptimizer(0.01)

        train_op = optimizer.minimize(loss)

        image_batches, label_batches = get_batches(images, labels, self.batch_size)
        print type(image_batches)
        print type(image_batches[0])
        with tf.Session() as sess:
            for i in range(epochs):
                loss_train = 0
                for j in range(num_batches):
                    _, loss_val = sess.run([train_op, loss],
                                           feed_dict={images: image_batches[j], labels: label_batches[j]})
                    loss_train += loss_val
                    print "Batch loss: {}".format(loss_val)

                print "\nTraining set loss: {}".format(loss_train)

        saver = tf.train.Saver()
        save_path = saver.save(sess, os.path.join(os.getcwd(), os.pardir, "saved_models",
                                                  "{}.ckpt".format(self.model_name)))
        print "Model saved at path {}".format(save_path)

    def evaluate(self, images, labels):
        """Evaluates the model on dev/test set."""

        num_batches = int(images.get_shape()[0] / self.batch_size)
        image_batches, label_batches = get_batches(images, labels, batch_size=self.batch_size)
        predictions = tf.argmax(self.logits, 1)
        accuracy = tf.metrics.accuracy(labels, predictions)
        acc = 0
        with tf.Session() as sess:
            for j in range(num_batches):
                acc_val = sess.run([accuracy], feed_dict={images: image_batches[j], labels: label_batches[j]})
                acc += acc_val

        print "Dev set accuracy: {}".format(acc/num_batches)
