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

train_paths = {
                "tabular" : select_split(tabular_data, train_idx),
                "bathroom" : select_split(bathroom_paths, train_idx),
                "bedroom" : select_split(bedroom_paths, train_idx),
                "kitchen" : select_split(kitchen_paths, train_idx),
                "frontal" : select_split(frontal_paths, train_idx)
}

train_labels = select_split(labels, train_idx)

test_paths = {
                "tabular" : select_split(tabular_data, test_idx),
                "bathroom" : select_split(bathroom_paths, test_idx),
                "bedroom" : select_split(bedroom_paths, test_idx),
                "kitchen" : select_split(kitchen_paths, test_idx),
                "frontal" : select_split(frontal_paths, test_idx)
}

test_labels = select_split(labels, test_idx)



