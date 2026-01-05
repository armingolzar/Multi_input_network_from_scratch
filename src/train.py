import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReaduceLROnPlateau, TensorBoard, CSVLogger)
from data_loader import data_prepration
from utils import training_curve
from model import HousePriceCusomModel, SmoothL1Loss


model = HousePriceCusomModel()

model.compile(
                loss=SmoothL1Loss(delta=1.0),
                optimizer=tf.keras.optimizer.Adam(learning_rate=1e-3),
                metrics=["mae"]
)


