
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

interpreter = tf.lite.Interpreter(
    model_path="../human_detect.tflite",
    num_threads=1,
    experimental_delegates=[],  # Disable all delegates
    experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()

def get_prediction(img):
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale = input_details['quantization'][0]
    input_zero = input_details['quantization'][1]

    output_scale = output_details['quantization'][0]
    output_zero = output_details['quantization'][1]

    img = (img / input_scale) + input_zero

    interpreter.set_tensor(input_details['index'], img.astype(np.int8))
    interpreter.invoke()

    out = interpreter.get_tensor(output_details['index']).astype(np.float32)
    out = (out - output_zero) * output_scale

    return out

img = img_to_array(load_img('../images/1.png', target_size=(224, 224)))
score = get_prediction(img)[0][0]
print(f"Image 1: Score = {score}")

img = img_to_array(load_img('../images/2.png', target_size=(224, 224)))
score = get_prediction(img)[0][0]
print(f"Image 2: Score = {score}")
