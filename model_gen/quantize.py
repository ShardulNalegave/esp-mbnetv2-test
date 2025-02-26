
import os
import numpy as np
import tensorflow as tf

def representative_data_gen():
    from tensorflow.keras.utils import load_img, img_to_array
    num_samples_each = 100
    file_names = \
        list(map(lambda x: f"dataset/0/{x}", sorted(os.listdir('dataset/0'))[:num_samples_each])) + \
        list(map(lambda x: f"dataset/1/{x}", sorted(os.listdir('dataset/1'))[:num_samples_each]))

    for img_file in file_names:
        img = load_img(img_file, target_size=(224, 224))
        img_array = img_to_array(img)

        # Generate multiple versions of each image
        variants = []

        # Original image
        img_array = img_array.astype(np.float32)
        img_normalized = img_array / 127.5 - 1
        variants.append(img_normalized)

        # Brightness variations
        for brightness in [0.8, 1.2]:
            variant = tf.image.adjust_brightness(img_array, brightness - 1)
            variant = variant / 127.5 - 1
            variants.append(variant.numpy())

        # Contrast variations
        for contrast in [0.8, 1.2]:
            variant = tf.image.adjust_contrast(img_array, contrast)
            variant = variant / 127.5 - 1
            variants.append(variant.numpy())

        for variant in variants:
            yield [np.expand_dims(variant, axis=0).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model("human_detect_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter._experimental_new_quantizer = True
converter._experimental_disable_per_channel = False
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting model to TFLite...")
tflite_model = converter.convert()

with open("human_detect.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Model exported to human_detect.tflite")