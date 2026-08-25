import onnx


INPUT = "sidewalk_tobias_hailo.onnx"
OUTPUT = "sidewalk_tobias_hailo_patched.onnx"


model = onnx.load(INPUT)

count = 0

for node in model.graph.node:

    if node.op_type != "Conv":
        continue

    # Kalau sudah punya kernel_shape, skip
    if any(attr.name == "kernel_shape" for attr in node.attribute):
        continue

    # Ambil weight Conv
    weight_name = node.input[1]

    initializer = next(
        (x for x in model.graph.initializer if x.name == weight_name),
        None
    )

    if initializer is None:
        print("WARNING: weight tidak ditemukan:", node.name)
        continue

    # ONNX Conv weight:
    # [out_channels, in_channels/groups, kernel_h, kernel_w]
    shape = initializer.dims

    if len(shape) != 4:
        print(
            "WARNING: weight bukan Conv2D:",
            node.name,
            shape
        )
        continue

    kernel_h = shape[2]
    kernel_w = shape[3]

    node.attribute.append(
        onnx.helper.make_attribute(
            "kernel_shape",
            [kernel_h, kernel_w]
        )
    )

    print(
        f"Patched: {node.name} "
        f"kernel_shape=[{kernel_h}, {kernel_w}]"
    )

    count += 1


onnx.save(model, OUTPUT)

print()
print("================================")
print("PATCH SELESAI")
print(f"Conv dipatch: {count}")
print(f"Output: {OUTPUT}")
print("================================")