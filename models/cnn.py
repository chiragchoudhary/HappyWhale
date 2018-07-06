from processing import *
tf.logging.set_verbosity(tf.logging.DEBUG)


class CNN:
    """Defines a Convolution Neural Network with 1 convolution layer, and 1 fully connected layer"""

    def __init__(self, batch_size=32):
        """Define the CNN model architecture."""

        self.model_name = "CNN"
        self.batch_size = batch_size
        self.training_mode = tf.placeholder_with_default(True, shape=())
        self.images = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
        self.labels = tf.placeholder(tf.float32, shape=[None, 4251])
        self.out = tf.layers.conv2d(self.images, 1, 3, strides=(2, 2), padding="same")
        self.out = tf.nn.relu(self.out)
        self.out = tf.layers.max_pooling2d(self.out, 2, 2)
        self.out = tf.layers.dropout(self.out, rate=0.15, training=self.training_mode)
        self.out = tf.layers.conv2d(self.out, 1, 3, strides=(2, 2), padding="same")
        self.out = tf.nn.relu(self.out)
        self.out = tf.layers.max_pooling2d(self.out, 2, 2)
        self.out = tf.reshape(self.out, [-1, 32 * 32 * 1])
        self.logits = tf.layers.dense(self.out, 4251, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    def init(self):
        """Initialize the model variables."""
        # TODO
        return

    def train(self, images, labels, epochs=10):
        """Trains the model on training data."""

        stops = 3
        min_eval_loss = 100000000
        num_batches = int(len(images)/self.batch_size)
        print "Number of batches per epoch: {}".format(num_batches)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        optimizer = tf.train.AdamOptimizer(0.001)

        train_op = optimizer.minimize(loss)

        sess = tf.Session()
        saver = tf.train.Saver()

        with sess:
            image_batches, label_batches = get_labeled_batches(images, labels, self.batch_size)
            sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(epochs):
                epoch_loss = 0
                for j in range(num_batches):
                    img_batch, labels_batch = sess.run([image_batches, label_batches])

                    _, loss_val = sess.run([train_op, loss],
                                           feed_dict={self.images: img_batch, self.labels: labels_batch})
                    epoch_loss += loss_val

                print "Average loss for epoch {0}: {1}".format(i+1, 1.0*epoch_loss/num_batches)

                # early stopping condition
                if i % 3 == 0:
                    eval_loss = self.evaluate(images, labels)
                    if eval_loss > min_eval_loss:
                        print "No Improvement!!"
                        stops += 1
                        if stops == 3:
                            break
                    else:
                        print "Improvement!!"
                        min_eval_loss = eval_loss
                        stops = 0

            save_path = saver.save(sess, os.path.join(os.getcwd(), "saved_models",
                                   "{}.ckpt".format(self.model_name)))

        print "Model saved at path {}".format(save_path)
        coord.request_stop()
        sess.close()

    def evaluate(self, images, labels):
        """Evaluates the model on dev set and returns the total loss."""

        num_batches = int(len(images) / self.batch_size)
        sess = tf.get_default_session()
        with sess.as_default():
            image_batches, label_batches = get_labeled_batches(images, labels, self.batch_size, shuffle=False,
                                                               num_epochs=1)
            sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            eval_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            total_loss = 0
            for j in range(num_batches):
                img_batch, labels_batch = sess.run([image_batches, label_batches])
                loss_value = sess.run(eval_loss, feed_dict={self.images: img_batch,
                                                            self.labels: labels_batch,
                                                            self.training_mode: False})
                total_loss += loss_value
                print j

        print "Dev set loss: {}".format(total_loss)
        coord.request_stop()

        return total_loss

    def write_test_output(self, images, k):
        """Computes predictions for new images, and stores top k results in a csv file."""

        saver = tf.train.Saver()
        ind = []
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(os.getcwd(), "saved_models", "{}.ckpt".format(self.model_name)))
            print "Model restore successful!!"

            image_batches = get_test_batches(images, 128, num_epochs=1)
            sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            top_k = tf.nn.top_k(self.logits, k=k, sorted=True)
            while True:
                try:
                    image_batch = sess.run(image_batches)
                    _, indices = sess.run(top_k, feed_dict={self.images: image_batch, self.training_mode: False})
                    ind.append(indices)
                except tf.errors.OutOfRangeError:
                    print "Test data set processed successfully!"
                    break
                except Exception as e:
                    print e

        coord.request_stop()
        sess.close()

        # Obtain top k predictions for this batch, and append to output.
        top_k_classes = get_top_k_classes(ind)

        # Write output to a file in Kaggle submission format.
        submission_df = pd.DataFrame(data={'Image': images, 'Id': top_k_classes})
        submission_df.to_csv(os.path.join(os.getcwd(), "data", "submission.csv"), columns=['Image', 'Id'], sep=',', index=False)



