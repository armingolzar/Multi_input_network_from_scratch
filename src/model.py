import tensorflow as tf

def image_branch(name):
    inp = tf.keras.Input(shape=(32, 32, 3), name=name)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return inp, x

tab_inp = tf.keras.Input(shape=(4,), name="tabular")
tab_x = tf.keras.layers.Dense(32, activation="relu")(tab_inp)

bath_inp, bath_x = image_branch("bathroom")
bed_inp, bed_x = image_branch("bedroom")
kit_inp, kit_x = image_branch("kitchen")
front_inp, front_x = image_branch("frontal")

x = tf.keras.layers.Concatenate()([tab_x, bath_x, bed_x, kit_x, front_x])
x = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(
    inputs=[tab_inp, bath_inp, bed_inp, kit_inp, front_inp],
    outputs=out
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="mse",
    metrics=["mae"]
)

model.summary()
