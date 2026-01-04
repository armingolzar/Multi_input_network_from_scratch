import tensorflow as tf 
from model import HousePriceCusomModel, SmoothL1Loss
from data_loader import data_prepration
from utils import training_curve_ctl

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

train_ds, test_ds = data_prepration()

history = {"loss": [], "val_loss": []}
best_val_accuracy = None

for epoch in range(EPOCH):
    print(f"\n Epoch {epoch + 1}/ {EPOCH}")

    train_loss_metric.reset_state()
    val_loss_metric.reset_state()

    for (image, tabular), labels in train_ds:
        train_step(image, tabular, labels)

    for (image, tabular), labels in test_ds:
        val_step(image, tabular, labels)

    history["loss"].append(train_loss_metric.result().numpy())
    history["val_loss"].append(val_loss_metric.result().numpy())
    
    if (best_val_accuracy is None) or best_val_accuracy > val_loss_metric.result().numpy():
        best_val_accuracy = val_loss_metric.result().numpy()
        model.save("multi_input_house_scratch")
        print("saved model with val_loss", best_val_accuracy)

    print(f"Train Loss: {train_loss_metric.result():.5f}", "|", f"Val Loss: {val_loss_metric.result():.5f}")

training_curve_ctl(history=history)