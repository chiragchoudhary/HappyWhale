import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from models.cnn import CNN

from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

image_dir = os.path.join(os.getcwd(), "data", "img")
labels_file = os.path.join(os.getcwd(), "data", "train.csv")

labels = pd.read_csv(labels_file, delimiter=',')

print "Number of unique whale species >= {}".format(labels['Id'].nunique())

label_encoder = LabelBinarizer()
label_encoder.fit(labels['Id'])

for i, label in enumerate(labels['Image']):
    labels.loc[i, 'Image'] = str(os.path.join(image_dir, labels.loc[i, 'Image']))

train_x, val_x = train_test_split(labels, test_size=0.2)

train_images = tf.convert_to_tensor(train_x['Image'].tolist(), dtype=tf.string)
train_labels = tf.convert_to_tensor(label_encoder.transform(train_x['Id'].tolist()))
val_images = tf.convert_to_tensor(val_x['Image'].tolist(), dtype=tf.string)
val_labels = tf.convert_to_tensor(label_encoder.transform(val_x['Id'].tolist()))


model = CNN()
model.init()
model.train(train_x['Image'].tolist(), label_encoder.transform(train_x['Id'].tolist()))
print "model training complete...."

model.evaluate(val_images, val_labels)
