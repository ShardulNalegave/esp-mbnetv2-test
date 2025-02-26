import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
cap = cv2.VideoCapture(0)
interpreter = tf.lite.Interpreter(
    model_path="human_detect.tflite",
    num_threads=1,
    experimental_delegates=[],  # Disable all delegates
    experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()

def get_prediction(img):
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

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = cv2.resize(rgb_frame, IMG_SIZE)

    input_image = resized_frame.astype(np.float32)
    input_image = preprocess_input(input_image)

    input_batch = np.expand_dims(input_image, axis=0)

    predictions = get_prediction(input_batch)
    human_detected = predictions[0][0] > 0.5

    label = "Human Detected" if human_detected else "No Human"
    color = (0, 255, 0) if human_detected else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
