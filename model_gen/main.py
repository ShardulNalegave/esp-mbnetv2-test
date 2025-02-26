
import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import image_dataset_from_directory

def get_model():
    base_model = tf.keras.applications.MobileNetV2(
        alpha=0.35,
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    return model

def preprocess(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

if __name__ == '__main__':
    BATCH_SIZE = 50
    IMG_SIZE = (224, 224)
    directory = "dataset/"
    train_dataset = image_dataset_from_directory(
        directory,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        validation_split=0.2,
        subset='training',
        seed=42,
    ).map(preprocess)
    validation_dataset = image_dataset_from_directory(
        directory,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=42,
    ).map(preprocess)

    model = get_model()

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    print("Training the classifier...")
    initial_epochs = 50
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

    base_model = model.layers[1]
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    fine_tune_at = 120

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = None
    
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * base_learning_rate)
    metrics = ['accuracy']
    model.compile(loss=loss_function, optimizer = optimizer, metrics=metrics)

    print(f"Fine-Tuning the base model (layer {fine_tune_at} onwards)...")

    fine_tune_epochs = 25
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
    )

    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    model.save('human_detect.keras')
    model.export('human_detect_saved_model')