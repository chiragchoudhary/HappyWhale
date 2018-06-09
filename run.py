import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from models.cnn import CNN

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

image_dir = os.path.join(os.getcwd(), "data", "img")
labels_file = os.path.join(os.getcwd(), "data", "train.csv")

labels = pd.read_csv(labels_file, delimiter=',')

print labels['Id'].nunique()

for i, label in enumerate(labels['Image']):
    labels.loc[i, 'Image'] = str(os.path.join(image_dir, labels.loc[i, 'Image']))

train_x, val_x = train_test_split(labels, test_size=0.2)

train_images = tf.convert_to_tensor([train_x['Image'].tolist()], dtype=tf.string)
train_labels = tf.convert_to_tensor([train_x['Id'].tolist()], dtype=tf.string)
val_images = tf.convert_to_tensor([val_x['Image'].tolist()], dtype=tf.string)
val_labels = tf.convert_to_tensor([val_x['Id'].tolist()], dtype=tf.string)

model = CNN()
model.init()
model.train(train_images, train_labels, epochs=2)
