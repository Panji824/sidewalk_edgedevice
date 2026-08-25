import onnx
from onnx import helper

INPUT = "sidewalk_tobias_hailo_patched.onnx"
OUTPUT = "sidewalk_tobias_matmul_patched.onnx"

model = onnx.load(INPUT)
graph = model.graph

target = None

for node in graph.node:
    if node.op_type == "MatMul" and node.name == "node_MatMul_240":
        target = node
        break

if target is None:
    print("ERROR: node_MatMul_240 tidak ditemukan")
    print("Daftar MatMul:")
    for node in graph.node:
        if node.op_type == "MatMul":
            print(node.name, list(node.input), list(node.output))
    raise SystemExit(1)

print("Target ditemukan:")
print("Name   :", target.name)
print("Inputs :", list(target.input))
print("Output :", list(target.output))

old_b = target.input[1]

transpose_name = old_b + "_hailo_transpose"
transpose_output = old_b + "_hailo_transposed"

transpose_node = helper.make_node(
    "Transpose",
    inputs=[old_b],
    outputs=[transpose_output],
    name="hailo_fix_matmul5_transpose",
    perm=[0, 1, 3, 2]
)

target.input[1] = transpose_output

# Insert transpose immediately before target MatMul
new_nodes = []

for node in graph.node:
    if node == target:
        new_nodes.append(transpose_node)
    new_nodes.append(node)

graph.ClearField("node")
graph.node.extend(new_nodes)

onnx.save(model, OUTPUT)

print()
print("================================")
print("MATMUL PATCH SELESAI")
print("================================")
print("Original :", INPUT)
print("Output   :", OUTPUT)
print("Patched  :", target.name)
print("Transpose:", old_b, "->", transpose_output)
print("Perm     : [0, 1, 3, 2]")
print("================================")
