import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("..\\models\\checkpoints\\best_model", compile=False)

tab_mean = np.load("..\\models\\stats\\tab_mean.npy")
tab_std = np.load("..\\models\\stats\\tab_std.npy")
label_min = np.load("..\\models\\stats\\label_min.npy")
label_range = np.load("..\\models\\stats\\label_range.npy")

