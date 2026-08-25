import os
import random
import numpy as np
from PIL import Image

SRC = "/mnt/d/sidewalk_road_dataset/training/images"
OUT = "calibration.npy"

NUM_IMAGES = 128
SIZE = (224, 224)

# Cari gambar
files = []
for root, _, names in os.walk(SRC):
    for name in names:
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            files.append(os.path.join(root, name))

print(f"Total images ditemukan: {len(files)}")

if len(files) < NUM_IMAGES:
    raise RuntimeError(f"Gambar hanya {len(files)}, butuh minimal {NUM_IMAGES}")

random.seed(42)
selected = random.sample(files, NUM_IMAGES)

calib = []

for idx, path in enumerate(selected):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(SIZE, Image.Resampling.BILINEAR)

        # RGB -> float32, 0..1
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))

        calib.append(arr)

        if (idx + 1) % 16 == 0:
            print(f"Processed {idx + 1}/{NUM_IMAGES}")

    except Exception as e:
        print(f"SKIP: {path} -> {e}")

calib = np.stack(calib, axis=0).astype(np.float32)

print("Shape:", calib.shape)
print("Dtype:", calib.dtype)
print("Min:", calib.min())
print("Max:", calib.max())

np.save(OUT, calib)

print("=" * 40)
print(f"Calibration selesai: {OUT}")
print("=" * 40)
