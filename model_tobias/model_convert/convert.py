import os
import torch
import onnx

from transformers import SegformerForSemanticSegmentation


# ============================================================
# PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = MODEL_DIR

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "output"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "sidewalk_tobias.onnx"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("========================================")
print("SEGFORMER → ONNX")
print("========================================")
print(f"Model  : {MODEL_PATH}")
print(f"Output : {OUTPUT_PATH}")
print("========================================")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SegFormer...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

print("Model berhasil dimuat.")


# ============================================================
# WRAPPER
# ============================================================

class SegformerWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(
            pixel_values=pixel_values
        )

        return outputs.logits


wrapped_model = SegformerWrapper(model)

wrapped_model.eval()


# ============================================================
# DUMMY INPUT
# ============================================================

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
    dtype=torch.float32
)


# ============================================================
# TEST FORWARD
# ============================================================

print("\nTesting PyTorch forward...")

with torch.no_grad():

    output = wrapped_model(
        dummy_input
    )

print("Input :", dummy_input.shape)
print("Output:", output.shape)


# ============================================================
# EXPORT ONNX
# ============================================================

print("\nExporting ONNX...")

torch.onnx.export(

    wrapped_model,

    (dummy_input,),

    OUTPUT_PATH,

    input_names=[
        "pixel_values"
    ],

    output_names=[
        "logits"
    ],

    opset_version=17,

    do_constant_folding=True
)


print("\nONNX berhasil dibuat:")
print(OUTPUT_PATH)


# ============================================================
# CHECK ONNX
# ============================================================

print("\nChecking ONNX...")

onnx_model = onnx.load(
    OUTPUT_PATH
)

onnx.checker.check_model(
    onnx_model
)

print("ONNX VALID.")


# ============================================================
# SHOW INPUT / OUTPUT
# ============================================================

print("\n========================================")
print("ONNX INFORMATION")
print("========================================")

for x in onnx_model.graph.input:

    shape = [
        d.dim_value
        for d in x.type.tensor_type.shape.dim
    ]

    print(
        "INPUT :",
        x.name,
        shape
    )


for x in onnx_model.graph.output:

    shape = [
        d.dim_value
        for d in x.type.tensor_type.shape.dim
    ]

    print(
        "OUTPUT:",
        x.name,
        shape
    )


print("\n========================================")
print("SELESAI")
print("========================================")