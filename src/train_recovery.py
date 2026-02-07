import tensorflow as tf
from recovery_model import HousePriceModel, CustomL1Smooth
from recovery_data import create_train_test_ds
from utils import training_curve_ctl

EPOCHS = 50

model = HousePriceModel()
loss_fn = CustomL1Smooth(delta=1.0)

optmizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_loss_metric = tf.keras.metrics.mean(name="val_loss")

