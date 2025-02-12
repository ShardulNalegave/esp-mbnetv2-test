
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

interpreter = tf.lite.Interpreter(model_path="mobilenet_v2_35_quantized.tflite", experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

img = img_to_array(load_img("bus.jpg", target_size=(224, 224)))
# img.astype(np.uint8).tofile("bus.raw")

img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
img = (img / input_scale) + input_zero
img = img.astype(np.int8)

print(img.flatten()[:100])
interpreter.set_tensor(input_details['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details['index']).astype(np.float32)
output = (output - output_zero) * output_scale
predictions = decode_predictions(output, top=5)
print(predictions)
