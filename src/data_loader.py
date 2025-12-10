import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob


house_tabular_data = pd.read_csv("..\\data\\house_dataset\\HousesInfo.txt", sep=" ", header=None, 
                                 names=["feature1", "feature2", "feature3", "feature4", "price"])

tabular_data = house_tabular_data.iloc[:, :-1].astype("float32")

labels = house_tabular_data.iloc[:, -1].astype("float32")


bathroom_paths = sorted(glob.glob("..\\data\\house_dataset\\*bathroom*.jpg"))
bedroom_paths = sorted(glob.glob("..\\data\\house_dataset\\*bedroom*.jpg"))
kitchen_paths = sorted(glob.glob("..\\data\\house_dataset\\*kitchen*.jpg"))
frontal_paths = sorted(glob.glob("..\\data\\house_dataset\\*frontal*.jpg"))


tabular_data = tf.constant(tabular_data)
labels = tf.constant(labels)
bathroom_paths = tf.constant(bathroom_paths)
bedroom_paths = tf.constant(bedroom_paths)
kitchen_paths = tf.constant(kitchen_paths)
frontal_paths = tf.constant(frontal_paths)
