import tensorflow as tf 
from model import HousePriceCusomModel, SmoothL1Loss
from data_loader import 

EPOCH = 50

model = HousePriceCusomModel()
loss_fn = SmoothL1Loss(delta=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

@tf.function
def train_step(image, tabular, labels):
    with tf.GradientTape() as tape:
        predictions = model((image, tabular), training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_metric.update_state(loss)

@tf.function
def val_step(image, tabular, labels):
    prediction = model((image, tabular), training=False)
    loss = loss_fn(labels, prediction)

    val_loss_metric.update_state(loss)

for epoch in range(EPOCH):
    print(f"\n Epoch {epoch + 1}/ {EPOCH}")

    train_loss_metric.reset_state()
    val_loss_metric.reset_state()

    for (image, tabular), labels in tra_ds