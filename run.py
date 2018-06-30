import os
from processing import get_train_val_split, get_jpeg_files

from models.cnn import CNN

labels_file = os.path.join(os.getcwd(), "data", "train.csv")
train_X, val_X, train_Y, val_Y = get_train_val_split(labels_file)

model = CNN()
model.init()
#model.train(train_X, train_Y)
print "model training complete!!"
print "Evaluating on validation data set.."

# model.evaluate(val_X, val_Y)

# Predict top k labels for test data images, and store in submission format
k = 5
test_images_dir = os.path.join(os.getcwd(), "data", "test")
test_images = get_jpeg_files(test_images_dir)

model.write_test_output(test_images, k)
