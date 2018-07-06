import os
from processing import get_train_val_split, get_jpeg_files

from models.cnn import CNN

labels_file = os.path.join(os.getcwd(), "data", "train.csv")
train_X, val_X, train_Y, val_Y = get_train_val_split(labels_file, split_ratio=0)

model = CNN()
model.init()
model.train(train_X, train_Y, val_X, val_Y, epochs=50)
print "model training complete!!"

# Predict top k labels for test data images, and store in submission format
k = 5
test_images_dir = os.path.join(os.getcwd(), "data", "test")
test_images = get_jpeg_files(test_images_dir)

model.write_test_output(test_images, k)
