import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from utils import preprocess_image_test, preprocess_image_train
from config import BUFFER_SIZE, BATCH_SIZE

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE

import tensorflow as tf
import os

trainA_dir = "/home/data-dynamo/AI_ML/Ghilbi/kaggle/input/real-to-ghibli-image-dataset-5k-paired-images/dataset/trainA"
trainB_dir = "/home/data-dynamo/AI_ML/Ghilbi/kaggle/input/real-to-ghibli-image-dataset-5k-paired-images/dataset/trainB_ghibli"

trainA_files = [os.path.join(trainA_dir, f) for f in os.listdir(trainA_dir) if f.endswith(('.jpg','.png','.jpeg'))]
trainB_files = [os.path.join(trainB_dir, f) for f in os.listdir(trainB_dir) if f.endswith(('.jpg','.png','.jpeg'))]

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    return img 


train_real = tf.data.Dataset.from_tensor_slices(trainA_files).map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)
train_ghibli = tf.data.Dataset.from_tensor_slices(trainB_files).map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)


train_real = train_real.cache().map(
    preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE
).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_ghibli = train_ghibli.cache().map(
    preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE
).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_real = train_real.map(
    preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE
).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_ghibli = train_ghibli.map(
    preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE
).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_real = next(iter(train_real))
sample_ghibli = next(iter(train_ghibli))