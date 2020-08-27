import tensorflow as tf
from tensorflow.keras import layers
import random

IMG_WIDTH = 192
IMG_HEIGHT = 192
DATA = "./data"
MODEL_PATH = "./cnn"
BATCH_SIZE = 30
NUM_CLASSES = 2
EPOCHS = 10
FINE_TUNE_EPOCHS = 20


def load_datasets():
    seed = random.randint(1, 1024)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATA}/train",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        seed=seed,
        validation_split=0.2,
        subset="training",
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATA}/train",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        seed=seed,
        validation_split=0.2,
        subset="validation",
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATA}/test", image_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=32
    )
    return train_ds, val_ds, test_ds


def create_model():
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return base_model, model


def train_model():
    train_ds, val_ds, test_ds = load_datasets()
    base_model, model = create_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    # Fine-tune from this layer onwards
    fine_tune_at = 100

    base_model.trainable = True
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=train_ds,
    )

    results = model.evaluate(test_ds)
    print(f"Test loss, test accuracy: {results}")
    model.save(MODEL_PATH)
