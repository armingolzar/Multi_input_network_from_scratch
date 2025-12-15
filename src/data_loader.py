import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob


house_tabular_data = pd.read_csv("..\\data\\house_dataset\\HousesInfo.txt", sep=" ", header=None, 
                                 names=["feature1", "feature2", "feature3", "feature4", "price"])

tabular_data = house_tabular_data.iloc[:, :-1].astype("float32")

labels = house_tabular_data.iloc[:, -1].astype("float32")


bathroom_paths = sorted(glob.glob("..\\data\\house_dataset\\*bathroom*.jpg"))
bedroom_paths = sorted(glob.glob("..\\data\\house_dataset\\*bedroom*.jpg"))
kitchen_paths = sorted(glob.glob("..\\data\\house_dataset\\*kitchen*.jpg"))
frontal_paths = sorted(glob.glob("..\\data\\house_dataset\\*frontal*.jpg"))

n_samples = len(labels)
assert all(len(x) == n_samples for x in [bathroom_paths, bedroom_paths, kitchen_paths, frontal_paths]), "Image count mismatch!"


tabular_data = tf.constant(tabular_data)
labels = tf.constant(labels)
bathroom_paths = tf.constant(bathroom_paths)
bedroom_paths = tf.constant(bedroom_paths)
kitchen_paths = tf.constant(kitchen_paths)
frontal_paths = tf.constant(frontal_paths)


train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42, shuffle=False)

def select_split(tensor, indices):
    return tf.gather(tensor, indices)

train_data = {
                "tabular" : select_split(tabular_data, train_idx),
                "bathroom" : select_split(bathroom_paths, train_idx),
                "bedroom" : select_split(bedroom_paths, train_idx),
                "kitchen" : select_split(kitchen_paths, train_idx),
                "frontal" : select_split(frontal_paths, train_idx)
}

train_labels = select_split(labels, train_idx)

test_data = {
                "tabular" : select_split(tabular_data, test_idx),
                "bathroom" : select_split(bathroom_paths, test_idx),
                "bedroom" : select_split(bedroom_paths, test_idx),
                "kitchen" : select_split(kitchen_paths, test_idx),
                "frontal" : select_split(frontal_paths, test_idx)
}

test_labels = select_split(labels, test_idx)

# Normalization

tab_mean = tf.reduce_mean(train_data["tabular"], axis=0)
tab_std = tf.math.reduce_std(train_data["tabular"], axis=0)
tab_std = tf.where(tab_std == 0, 1.0, tab_std)

label_min = tf.reduce_min(train_labels)
label_max = tf.reduce_max(train_labels)
label_range = tf.where(label_max - label_min == 0, 1.0, label_max - label_min)

# Preprocessing Functions

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (32, 32))
    return tf.cast(img, tf.float32) / 255.0

def prerocess(tabular, bath, bed, kitch, front, label):
    bath = load_image(bath)
    bed = load_image(bed)
    kitch = load_image(kitch)
    front = load_image(front)

    tabular = (tabular - tab_mean) / (tab_std + 1e-8)
    label = (label - label_min) / (label_range + 1e-8)

    return {"tabular" : tabular, "bathroom" : bath, "bedroom" : bed, "kitchen" : kitch, "frontal" : front}, label



