import os

import tensorflow as tf
from processing import get_batches
#tf.logging.set_verbosity(tf.logging.DEBUG)


class CNN:
    """Defines a Convolution Neural Network with 1 convolution layer, and 1 fully connected layer"""

    def __init__(self, batch_size=32):
        """Define the CNN model architecture."""

        self.model_name = "CNN"

        self.batch_size = batch_size
        self.images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.labels = tf.placeholder(tf.float32, shape=[None, 4251])
        self.out = tf.layers.conv2d(self.images, 16, 3, strides=(2, 2), padding="same")
        self.out = tf.nn.relu(self.out)
        self.out = tf.layers.max_pooling2d(self.out, 2, 2)
        self.out = tf.reshape(self.out, [-1, 16 * 16 * 16])
        self.logits = tf.layers.dense(self.out, 4251)

    def init(self):
        """Initialize the model variables."""
        # TODO
        return

    def train(self, images, labels, epochs=50):
        """Trains the model on training data."""
        num_batches = 20
        # num_batches = int(len(images)/self.batch_size)
        print "Number of batches per epoch: {}".format(num_batches)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        optimizer = tf.train.AdamOptimizer(0.001)

        train_op = optimizer.minimize(loss)

        sess = tf.Session()
        saver = tf.train.Saver()

        with sess:
            image_batches, label_batches = get_batches(images, labels, self.batch_size)
            sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            loss_val = 0
            for i in range(epochs):
                for j in range(num_batches):
                    img_batch, labels_batch = sess.run([image_batches, label_batches])

                    _, loss_val = sess.run([train_op, loss],
                                           feed_dict={self.images: img_batch, self.labels: labels_batch})

                print "Average loss for epoch {0}: {1}".format(i+1, loss_val/num_batches)

            save_path = saver.save(sess, os.path.join(os.getcwd(), "saved_models",
                                   "{}.ckpt".format(self.model_name)))

        print "Model saved at path {}".format(save_path)
        coord.request_stop()
        sess.close()

    def evaluate(self, images, labels):
        """Evaluates the model on dev/test set."""

        num_batches = 20  # int(images.get_shape()[0] / self.batch_size)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(os.getcwd(), "saved_models", "{}.ckpt".format(self.model_name)))
            print "Model restore successful!!"
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(v)
            for v in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES):
                print(v)

            image_batches, label_batches = get_batches(images, labels, self.batch_size, shuffle=False)

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)

            actual_labels = tf.argmax(label_batches, axis=1)
            predictions = tf.argmax(self.logits, 1)
            accuracy, _ = tf.metrics.accuracy(actual_labels, predictions, name="my_metric")
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)
            acc = 0
            for j in range(num_batches):
                img_batch, labels_batch = sess.run([image_batches, label_batches])
                pred, acc_val = sess.run([predictions, accuracy], feed_dict={self.images: img_batch, self.labels: labels_batch})
                acc += acc_val

        print "Dev set accuracy: {}".format((1.0*acc)/num_batches)
        coord.request_stop()
        sess.close()
