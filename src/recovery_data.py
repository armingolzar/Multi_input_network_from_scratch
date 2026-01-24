import tensorflow as tf 
import glob 
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import re

df = pd.read_csv("..\\data\\house_dataset\\HousesInfo.txt", sep=" ", header=None, names=["F1", "F2", "F3", "F4", "Label"])

tabular_data = df.iloc[:, :-1]
label = df.iloc[:, -1]

def natural_sort(file_list):
    return sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))

bathroom_addr = natural_sort(glob.glob("..\\data\\house_dataset\\*bathroom*.jpg"))
bedroom_addr = natural_sort(glob.glob("..\\data\\house_dataset\\*bedroom*.jpg"))
kitchen_addr = natural_sort(glob.glob("..\\data\\house_dataset\\*kitchen*.jpg"))
frontal_addr = natural_sort(glob.glob("..\\data\\house_dataset\\*frontal*.jpg"))

tabular_data = tf.constant(tabular_data, dtype=tf.float32)
label = tf.constant(label, dtype=tf.float32)
bathroom_addr = tf.constant(bathroom_addr)
bedroom_addr = tf.constant(bedroom_addr)
kitchen_addr = tf.constant(kitchen_addr)
frontal_addr = tf.constant(frontal_addr)

n_samples = len(label)
assert all([len(x) == n_samples for x in [bathroom_addr, bedroom_addr, kitchen_addr, frontal_addr]]), "Shape mismatch error"


all_indexs = range(len(label))
train_indexs, test_indexs = train_test_split(all_indexs, test_size=0.2, random_state=43, shuffle=False)

def split_data(data, index):
    splited_data = tf.gather(data, index)
    return splited_data

train_tabular = split_data(tabular_data, train_indexs)
train_label = split_data(label, train_indexs)
train_bathroom = split_data(bathroom_addr, train_indexs)
train_bedroom = split_data(bedroom_addr, train_indexs)
train_kitchen = split_data(kitchen_addr, train_indexs)
train_frontal = split_data(frontal_addr, train_indexs)

test_tabular = split_data(tabular_data, test_indexs)
test_label = split_data(label, test_indexs)
test_bathroom = split_data(bathroom_addr, test_indexs)
test_bedroom = split_data(bedroom_addr, test_indexs)
test_kitchen = split_data(kitchen_addr, test_indexs)
test_frontal = split_data(frontal_addr, test_indexs)

tab_mean = tf.reduce_mean(train_tabular, axis=0)
tab_std = tf.math.reduce_std(train_tabular, axis=0)
tab_std = tf.where(tab_std == 0, 1.0, tab_std)

label_min = tf.reduce_min(train_label)
label_max = tf.reduce_max(train_label)
label_range = tf.where(label_max - label_min == 0, 1.0, label_max - label_min)

np.save("..\\models\\stats\\tab_mean.npy", tab_mean.numpy())
np.save("..\\models\\stats\\tab_std.npy", tab_std.numpy())

np.save("..\\models\\stats\\label_min.npy", label_min.numpy())
np.save("..\\models\\stats\\label_range.npy", label_range.numpy())


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    return tf.cast(img, dtype=tf.float32) / 255.0

def preprocess(tabular, bath, bed, kitch, front, label):

    img_bath = load_img(bath)
    img_bed = load_img(bed)
    img_kitch = load_img(kitch)
    img_front = load_img(front)

    img_concat = tf.concat([img_bath, img_bed, img_kitch, img_front], axis=-1)

    tabular_norm = (tabular - tab_mean)/(tab_std + 1e-8)
    label = (label - label_min)/ (label_range + 1e-8)

    return ((img_concat, tabular_norm), label)

def create_train_test_ds():


    train_dataset = tf.data.Dataset.from_tensor_slices((train_tabular, train_bathroom, train_bedroom, train_kitchen, train_frontal, train_label))
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(512, seed=43).batch(32).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_tabular, test_bathroom, test_bedroom, test_kitchen, test_frontal, test_label))
    test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset



