
This folder contains the scripts used to get the human_detect.tflite model used in this example.

1. `main.py`: Create, train and fine-tune the model using MobileNetV2 as the base-model.
2. `quantize.py`: Quantize the generated model
3. `test.py`: An simple OpenCV app to test the model using Webcam input.

**Note:** To generate the model, first download and extract the https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset dataset into dataset/ folder
