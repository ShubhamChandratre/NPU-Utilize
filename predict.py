from openvino.runtime import Core
import cv2
import numpy as np

# Initialize OpenVINO Core
core = Core()

# Enable logging for supported devices
for device in core.available_devices:
    try:
        core.set_property(device, {"LOG_LEVEL": "LOG_INFO"})
    except Exception as e:
        print(f"[WARN] Could not set LOG_LEVEL for {device}: {e}")

# Load model
model_path = "C:/Users/SHUBHAM/Desktop/public/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml"
model = core.read_model(model_path)

# Compile model for NPU (will raise error if not compatible)
try:
    compiled_model = core.compile_model(model, "NPU")
except Exception as e:
    print(f"❌ Could not compile model for NPU: {e}")
    print("🔁 Falling back to CPU...")
    compiled_model = core.compile_model(model, "CPU")

# Print which device is being used
device_used = compiled_model.get_property("EXECUTION_DEVICES")
print(f"✅ Model is running on: {device_used}")

# Get input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load and preprocess image
image = cv2.imread("dog2.jpg")
if image is None:
    print("❌ Error: 'dog2.jpg' not found.")
    exit()

# Step 1: Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Resize to 224x224
resized_image = cv2.resize(image, (224, 224))

# Step 3: Normalize using ImageNet mean/std
input_image = resized_image.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
input_image = (input_image - mean) / std

# Step 4: HWC → CHW → NCHW
input_image = input_image.transpose((2, 0, 1))  # CHW
input_image = np.expand_dims(input_image, axis=0)  # NCHW
input_image = input_image.astype(np.float32)


# Run inference
result = compiled_model([input_image])[output_layer]
top_class = int(np.argmax(result))
print(f"🔢 Predicted class index: {top_class}")

# Load labels and show predicted class
try:
    with open("imagenet_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"🏷 Predicted label: {labels[top_class]}")
except FileNotFoundError:
    print("⚠️ 'imagenet_labels.txt' not found. Download from:")
    print("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
