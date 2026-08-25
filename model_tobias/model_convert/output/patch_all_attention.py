import onnx
from onnx import helper

INPUT = "sidewalk_tobias_hailo_patched.onnx"
OUTPUT = "sidewalk_tobias_attention_all_patched.onnx"

model = onnx.load(INPUT)
graph = model.graph

# MatMul attention Q @ K
targets = [
    "node_MatMul_240",
    "node_MatMul_324",
    "node_MatMul_419",
    "node_MatMul_503",
    "node_MatMul_587",
    "node_MatMul_660",
]

patched = []

new_nodes = []

for node in graph.node:

    if node.name in targets and node.op_type == "MatMul":

        original_k = node.input[1]

        transpose_name = original_k + "_attention_transposed"

        transpose_node = helper.make_node(
            "Transpose",
            inputs=[original_k],
            outputs=[transpose_name],
            name=transpose_name,
            perm=[0, 1, 3, 2],
        )

        node.input[1] = transpose_name

        new_nodes.append(transpose_node)
        patched.append(node.name)

    new_nodes.append(node)

graph.ClearField("node")
graph.node.extend(new_nodes)

onnx.checker.check_model(model)
onnx.save(model, OUTPUT)

print("================================")
print("ATTENTION PATCH SELESAI")
print("================================")
print("Input :", INPUT)
print("Output:", OUTPUT)
print("Patched:", patched)
print("ONNX VALID")
print("================================")
