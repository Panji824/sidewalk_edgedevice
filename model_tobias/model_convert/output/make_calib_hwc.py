import os
import glob
import cv2
import numpy as np

IMAGE_DIR = "/mnt/d/sidewalk_road_dataset/training/images"
OUT = "calibration_hwc.npy"

images = []
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

for ext in extensions:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

print(f"Total images ditemukan: {len(images)}")

# Ambil maksimal 128 gambar
images = images[:128]

calib = []

for i, path in enumerate(images):
    img = cv2.imread(path)

    if img is None:
        continue

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 224x224 sesuai input model
    img = cv2.resize(img, (224, 224))

    # float32 0..1
    img = img.astype(np.float32) / 255.0

    # HWC -> JANGAN transpose
    calib.append(img)

    if (i + 1) % 16 == 0:
        print(f"Processed {i + 1}/{len(images)}")

calib = np.stack(calib, axis=0).astype(np.float32)

print("Shape:", calib.shape)
print("Dtype:", calib.dtype)
print("Min:", calib.min())
print("Max:", calib.max())

np.save(OUT, calib)

print("=" * 40)
print(f"Calibration selesai: {OUT}")
print("=" * 40)
