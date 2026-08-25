import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import SegformerImageProcessor


# ============================================================
# PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.dirname(SCRIPT_DIR)

ONNX_PATH = os.path.join(
    SCRIPT_DIR,
    "output",
    "sidewalk_tobias.onnx"
)

IMAGE_PATH = "sidewalk_outsource/test_data/test_data_belok.png"

# ============================================================
# LOAD PROCESSOR
# ============================================================

processor = SegformerImageProcessor.from_pretrained(
    MODEL_DIR
)


# ============================================================
# LOAD ONNX
# ============================================================

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX berhasil dimuat.")

print("\nInput:")
for inp in session.get_inputs():
    print(
        inp.name,
        inp.shape,
        inp.type
    )

print("\nOutput:")
for out in session.get_outputs():
    print(
        out.name,
        out.shape,
        out.type
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# PREPROCESS
# ============================================================

inputs = processor(
    images=image,
    size={
        "height": 224,
        "width": 224
    },
    return_tensors="np"
)

pixel_values = inputs["pixel_values"].astype(
    np.float32
)


# ============================================================
# ONNX INFERENCE
# ============================================================

outputs = session.run(
    ["logits"],
    {
        "pixel_values": pixel_values
    }
)

logits = outputs[0]


print("\nLogits shape:")
print(logits.shape)


# ============================================================
# SEGMENTATION
# ============================================================

prediction = np.argmax(
    logits,
    axis=1
)

prediction = prediction[0]


print("\nPrediction shape:")
print(prediction.shape)


# ============================================================
# CHECK CLASS
# ============================================================

SIDEWALK_CLASS = 2

sidewalk_pixels = np.sum(
    prediction == SIDEWALK_CLASS
)

total_pixels = prediction.size

sidewalk_percentage = (
    sidewalk_pixels /
    total_pixels
) * 100


print("\n================================")
print("SIDEWALK RESULT")
print("================================")

print(
    f"Sidewalk pixels : {sidewalk_pixels}"
)

print(
    f"Total pixels    : {total_pixels}"
)

print(
    f"Sidewalk        : {sidewalk_percentage:.2f}%"
)