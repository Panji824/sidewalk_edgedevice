import os
import torch
import onnx

from transformers import SegformerForSemanticSegmentation


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "sidewalk_tobias_hailo.onnx"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("SEGFORMER → HAILO ONNX EXPORT")
print("========================================")

print(f"Model : {MODEL_DIR}")
print(f"Output: {OUTPUT_PATH}")

print("\nLoading model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.eval()

print("Model loaded.")
print(f"Hidden sizes : {model.config.hidden_sizes}")
print(f"Heads        : {model.config.num_attention_heads}")
print(f"SR ratios    : {model.config.sr_ratios}")


# ============================================================
# WRAPPER
# ============================================================

class SegformerHailoWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, pixel_values):

        outputs = self.model(
            pixel_values=pixel_values
        )

        return outputs.logits


wrapped_model = SegformerHailoWrapper(model)
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

print("\nTesting forward pass...")

with torch.no_grad():

    test_output = wrapped_model(
        dummy_input
    )

print("Forward OK.")
print("Output shape:", tuple(test_output.shape))


# ============================================================
# EXPORT
# ============================================================

print("\nExporting ONNX...")
print("This may take a while.")


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

    do_constant_folding=True,

    dynamo=True
)


print("\nONNX export completed.")


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
# INFORMATION
# ============================================================

print("\n========================================")
print("ONNX INFORMATION")
print("========================================")

for inp in onnx_model.graph.input:

    shape = []

    for dim in inp.type.tensor_type.shape.dim:

        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("?")

    print(
        "INPUT :",
        inp.name,
        shape
    )


for out in onnx_model.graph.output:

    shape = []

    for dim in out.type.tensor_type.shape.dim:

        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("?")

    print(
        "OUTPUT:",
        out.name,
        shape
    )


print("\n========================================")
print("SELESAI")
print("========================================")