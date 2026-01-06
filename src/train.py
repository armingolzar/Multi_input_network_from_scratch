import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger)
from data_loader import data_prepration
from utils import training_curve
from model import HousePriceCusomModel, SmoothL1Loss

callbacks = [ModelCheckpoint(filepath="..\\models\\checkpoints\\best_model", monitor="val_loss", save_best_only=True, save_format="tf", verbose=1), EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,
                            verbose=1), ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6, verbose=1), CSVLogger(filename="..\\models\\log\\train_log.csv", append=False)]

model = HousePriceCusomModel()
train_ds, val_ds = data_prepration()

model.compile(
                loss=SmoothL1Loss(delta=1.0),
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                metrics=["mae"]
)

# dummy_image = tf.zeros((1, 32, 32, 12))
# dummy_tabular = tf.zeros((1, 4))
# model((dummy_image, dummy_tabular))
# model.summary()

history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks, verbose=1)

training_curve(history)


