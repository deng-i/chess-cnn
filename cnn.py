import numpy as np
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# data_dir = pathlib.Path("C:/Users/hogyv/Downloads/Chess_ID_Public_Data/output_train")
# image_count = len(list(data_dir.glob("*/*.jpg")))

batch_size = 32
img_height = 180
img_width = 180
epochs = 15


def load_data(path):
    data_dir = pathlib.Path(path)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds, val_ds, class_names


def prepare_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def make_model(class_names, train_ds, val_ds):
    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history, model


def show_stats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def save_model(model):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


def load_model():
    TF_MODEL_FILE_PATH = 'model.tflite'
    try:
        interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    except ValueError:
        return None
    classify_lite = interpreter.get_signature_runner('serving_default')
    return classify_lite


def test_on_img(path, classify_lite, class_names):
    img_path = pathlib.Path(path)
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions_lite = classify_lite(sequential_input=img_array)['outputs']
    score_lite = tf.nn.softmax(predictions_lite)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    )


def predict_img(dir_path="C:/Users/hogyv/Downloads/Chess_ID_Public_Data/output_train", img_path="C:/Users/hogyv/Downloads/Chess_ID_Public_Data/output_test/bb/0761_2.jpg"):
    if load_model() is None:
        train_ds, val_ds, class_names = load_data(dir_path)
        train_ds, val_ds = prepare_data(train_ds, val_ds)
        history, model = make_model(class_names, train_ds, val_ds)
        show_stats(history)
        save_model(model)
        loaded_model = load_model()
        test_on_img(img_path, loaded_model, class_names)
    else:
        _, _, class_names = load_data(dir_path)
        loaded_model = load_model()
        test_on_img(img_path, loaded_model, class_names)

# predict_img()

