from openvino.runtime import Core
import cv2
import numpy as np

# Step 1: Load OpenVINO
core = Core()

# Step 2: Read the model
# model = core.read_model("mobilenet-v2-pytorch.xml")
model = core.read_model("C:/Users/SHUBHAM/Desktop/public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml")
compiled_model = core.compile_model(model, "AUTO")  # AUTO selects NPU if available

# Step 3: Get input and output layer names
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Step 4: Load and preprocess image
image = cv2.imread("snake.jpg")
resized_image = cv2.resize(image, (224, 224))  # MobileNetV2 expects 224x224
input_image = resized_image.transpose((2, 0, 1))  # HWC to CHW
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype(np.float32) / 255.0

# Step 5: Run inference
result = compiled_model([input_image])[output_layer]

# Step 6: Get top prediction
top_class = np.argmax(result)
print(f"Predicted class index: {top_class}")

# core.set_property("NPU", {"LOG_LEVEL": "LOG_INFO"})  # Verbose logging

# for device in core.available_devices:
#     core.set_property(device, {"LOG_LEVEL": "LOG_INFO"})

device_used = compiled_model.get_property("EXECUTION_DEVICES")
print(f"✅ Model is running on: {device_used}")