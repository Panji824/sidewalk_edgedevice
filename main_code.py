# ============================================================
# SIDEWALK EDGE DEVICE
# Raspberry Pi 5
# SegFormer + ONNX Runtime
# ============================================================


# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import cv2
import numpy as np
import onnxruntime as ort
import pygame
import os
import time


# ============================================================
# 2. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_PATH = "./model_tobias/sidewalk_tobias.onnx"


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ------------------------------------------------------------
# Model input
# ------------------------------------------------------------

MODEL_WIDTH = 224
MODEL_HEIGHT = 224


# ------------------------------------------------------------
# Class SegFormer
# ------------------------------------------------------------

UNLABELED_CLASS = 0
SIDEWALK_CLASS = 2
PERSON_CLASS = 8


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

NUM_BANDS = 6
NUM_COLUMNS = 5


# ------------------------------------------------------------
# Decision threshold
# ------------------------------------------------------------

B4_THRESHOLD = 2.0


# ------------------------------------------------------------
# Person detection
# ------------------------------------------------------------

MIN_PERSON_AREA = 100


# ------------------------------------------------------------
# Audio
# ------------------------------------------------------------

AUDIO_DIR = "./audio"

AUDIO_JALAN_TERUS = os.path.join(
    AUDIO_DIR,
    "audio_jalan_terus.wav"
)

AUDIO_PELAN = os.path.join(
    AUDIO_DIR,
    "audio_pelan_pelan_tidak_ada_sidewalk.wav"
)

AUDIO_BELok_KANAN = os.path.join(
    AUDIO_DIR,
    "audio_coba_belok_kanan.wav"
)

AUDIO_BELOK_KIRI = os.path.join(
    AUDIO_DIR,
    "audio_coba_belok_kiri_180.wav"
)


# ============================================================
# 3. AUDIO INITIALIZATION
# ============================================================

print("=" * 60)
print("INITIALIZING AUDIO")
print("=" * 60)

pygame.mixer.init()

print("Audio system initialized.")


# ============================================================
# 4. AUDIO FUNCTION
# ============================================================

def play_audio(audio_path):

    """
    Memutar file audio sesuai keputusan sistem.
    """

    if not os.path.exists(audio_path):

        print(
            "[AUDIO] File tidak ditemukan:",
            audio_path
        )

        return

    try:

        pygame.mixer.music.load(
            audio_path
        )

        pygame.mixer.music.play()

        print(
            "[AUDIO] Playing:",
            os.path.basename(audio_path)
        )

    except Exception as e:

        print(
            "[AUDIO ERROR]",
            e
        )


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

input_info = session.get_inputs()[0]

output_info = session.get_outputs()[0]


INPUT_NAME = input_info.name

OUTPUT_NAME = output_info.name


print(
    "[MODEL] Input name :",
    INPUT_NAME
)

print(
    "[MODEL] Input shape:",
    input_info.shape
)

print(
    "[MODEL] Output name:",
    OUTPUT_NAME
)

print(
    "[MODEL] Output shape:",
    output_info.shape
)

print(
    "[MODEL] Provider:",
    session.get_providers()
)


# ============================================================
# 6. PREPROCESSING IMAGE
# ============================================================

def preprocess(frame):

    """
    Mengubah frame kamera menjadi format
    yang sesuai dengan input model.
    """

    # BGR -> RGB

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Resize

    image = cv2.resize(
        image,
        (
            MODEL_WIDTH,
            MODEL_HEIGHT
        )
    )


    # Float32

    image = image.astype(
        np.float32
    )


    # Normalisasi ImageNet

    image = image / 255.0


    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )


    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )


    image = (
        image - mean
    ) / std


    # HWC -> CHW

    image = np.transpose(
        image,
        (
            2,
            0,
            1
        )
    )


    # Tambahkan batch

    image = np.expand_dims(
        image,
        axis=0
    )


    return image.astype(
        np.float32
    )


# ============================================================
# 7. SEGMENTATION
# ============================================================

def run_segmentation(frame):

    """
    Menjalankan inferensi SegFormer
    menggunakan ONNX Runtime.
    """

    input_tensor = preprocess(
        frame
    )


    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME:
            input_tensor
        }
    )


    logits = outputs[0]


    # --------------------------------------------------------
    # Output:
    # [1, class, height, width]
    # --------------------------------------------------------

    prediction = np.argmax(
        logits,
        axis=1
    )[0]


    # Resize segmentation ke ukuran frame

    prediction = cv2.resize(
        prediction.astype(
            np.uint8
        ),
        (
            frame.shape[1],
            frame.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )


    return prediction


# ============================================================
# 8. CREATE SIDEWALK MASK
# ============================================================

def create_sidewalk_mask(
    prediction
):

    """
    Membuat mask khusus sidewalk.
    """

    mask = (
        prediction ==
        SIDEWALK_CLASS
    ).astype(
        np.uint8
    )


    return mask


# ============================================================
# 9. CREATE PERSON MASK
# ============================================================

def create_person_mask(
    prediction
):

    """
    Membuat mask khusus manusia.
    """

    mask = (
        prediction ==
        PERSON_CLASS
    ).astype(
        np.uint8
    )


    return mask


# ============================================================
# 10. GRID BANDS
# ============================================================

def calculate_bands(
    sidewalk_mask
):

    """
    Membagi gambar menjadi 6 band horizontal.

    B1 = bagian paling jauh
    B6 = bagian paling dekat
    """

    height, width = (
        sidewalk_mask.shape
    )


    band_height = (
        height /
        NUM_BANDS
    )


    percentages = []


    for i in range(
        NUM_BANDS
    ):

        y_start = int(
            i *
            band_height
        )


        if i == NUM_BANDS - 1:

            y_end = height

        else:

            y_end = int(
                (i + 1) *
                band_height
            )


        band = sidewalk_mask[
            y_start:y_end,
            :
        ]


        sidewalk_pixels = np.sum(
            band == 1
        )


        total_pixels = band.size


        percentage = (
            sidewalk_pixels /
            total_pixels
        ) * 100


        percentages.append(
            percentage
        )


    return percentages


# ============================================================
# 11. DRAW BAND GRID
# ============================================================

def draw_bands(
    frame
):

    """
    Menggambar 6 garis horizontal
    pada frame.
    """

    height, width = (
        frame.shape[:2]
    )


    band_height = (
        height /
        NUM_BANDS
    )


    for i in range(
        1,
        NUM_BANDS
    ):

        y = int(
            i *
            band_height
        )


        cv2.line(
            frame,
            (0, y),
            (width, y),
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"B{i}",
            (
                10,
                y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


    return frame


# ============================================================
# 12. GRID COLUMNS
# ============================================================

def draw_columns(
    frame
):

    """
    Membagi gambar menjadi 5 kolom.

    C1 = kiri
    C3 = tengah
    C5 = kanan
    """

    height, width = (
        frame.shape[:2]
    )


    column_width = (
        width /
        NUM_COLUMNS
    )


    for i in range(
        1,
        NUM_COLUMNS
    ):

        x = int(
            i *
            column_width
        )


        cv2.line(
            frame,
            (x, 0),
            (x, height),
            (255, 255, 0),
            2
        )


    for i in range(
        NUM_COLUMNS
    ):

        x = int(
            i *
            column_width
        )


        cv2.putText(
            frame,
            f"C{i + 1}",
            (
                x + 10,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


    return frame


# ============================================================
# 13. PERSON + CENTROID
# ============================================================

def detect_person(
    person_mask,
    frame
):

    """
    Mendeteksi objek manusia menggunakan
    connected components.

    Centroid digunakan untuk menentukan
    posisi manusia pada kolom C1-C5.
    """

    height, width = (
        person_mask.shape
    )


    column_width = (
        width /
        NUM_COLUMNS
    )


    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            person_mask,
            connectivity=8
        )
    )


    person_count = 0

    person_columns = []


    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]


        # Abaikan objek kecil

        if area < MIN_PERSON_AREA:

            continue


        person_count += 1


        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        cx, cy = centroids[
            label_id
        ]


        # ----------------------------------------------------
        # Tentukan kolom
        # ----------------------------------------------------

        column = int(
            cx /
            column_width
        )


        column = min(
            column,
            NUM_COLUMNS - 1
        )


        column_number = (
            column + 1
        )


        person_columns.append(
            column_number
        )


        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        x = stats[
            label_id,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label_id,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label_id,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            label_id,
            cv2.CC_STAT_HEIGHT
        ]


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (
                int(cx),
                int(cy)
            ),
            7,
            (255, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Person C{column_number}",
            (
                x,
                max(
                    y - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )


    return (
        person_count,
        person_columns
    )


# ============================================================
# 14. DECISION MAKING
# ============================================================

def decision_making(
    band_percentages,
    person_columns
):

    """
    Menentukan keputusan alat berdasarkan:

    1. Kondisi sidewalk pada B4
    2. Keberadaan manusia pada C3
    """


    # --------------------------------------------------------
    # B4
    # --------------------------------------------------------

    B4 = band_percentages[3]


    # --------------------------------------------------------
    # Person di tengah
    # --------------------------------------------------------

    person_in_middle = (
        3 in person_columns
    )


    # --------------------------------------------------------
    # Jalan masih lurus
    # --------------------------------------------------------

    jalan_masih_lurus = (
        B4 > B4_THRESHOLD
    )


    # ========================================================
    # PRIORITAS ORANG
    # ========================================================

    if person_in_middle:

        decision = (
            "ADA ORANG DI TENGAH"
        )


        audio = AUDIO_PELAN


    # ========================================================
    # SIDEWALK MASIH ADA
    # ========================================================

    elif jalan_masih_lurus:

        decision = (
            "JALAN MASIH LURUS"
        )


        audio = AUDIO_JALAN_TERUS


    # ========================================================
    # SIDEWALK HABIS
    # ========================================================

    else:

        decision = (
            "SIDEWALK BERAKHIR"
        )


        audio = AUDIO_BELOK_KIRI


    return (
        decision,
        audio,
        jalan_masih_lurus,
        person_in_middle
    )


# ============================================================
# 15. DRAW SEGMENTATION
# ============================================================

def draw_segmentation(
    frame,
    sidewalk_mask,
    person_mask
):

    """
    Menampilkan hasil segmentasi:

    Hijau = sidewalk
    Merah = person
    """

    overlay = frame.copy()


    # Sidewalk

    overlay[
        sidewalk_mask == 1
    ] = (
        0,
        255,
        0
    )


    # Person

    overlay[
        person_mask == 1
    ] = (
        0,
        0,
        255
    )


    result = cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.3,
        0
    )


    return result


# ============================================================
# 16. CAMERA INITIALIZATION
# ============================================================

print("=" * 60)
print("STARTING CAMERA")
print("=" * 60)


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not cap.isOpened():

    raise RuntimeError(
        "Camera gagal dibuka."
    )


print(
    "Camera started."
)


# ============================================================
# 17. MAIN STREAMING LOOP
# ============================================================

print("=" * 60)
print("SIDEWALK SYSTEM RUNNING")
print("=" * 60)


last_audio = None

last_audio_time = 0

AUDIO_INTERVAL = 3.0


while True:

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print(
            "Gagal membaca frame."
        )

        break


    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    prediction = run_segmentation(
        frame
    )


    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    sidewalk_mask = (
        prediction ==
        SIDEWALK_CLASS
    ).astype(
        np.uint8
    )


    person_mask = (
        prediction ==
        PERSON_CLASS
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Draw segmentation
    # --------------------------------------------------------

    result = draw_segmentation(
        frame,
        sidewalk_mask,
        person_mask
    )


    # --------------------------------------------------------
    # Band analysis
    # --------------------------------------------------------

    band_percentages = (
        calculate_bands(
            sidewalk_mask
        )
    )


    # --------------------------------------------------------
    # Person detection
    # --------------------------------------------------------

    person_count, person_columns = (
        detect_person(
            person_mask,
            result
        )
    )


    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    result = draw_bands(
        result
    )


    result = draw_columns(
        result
    )


    # ========================================================
    # DECISION MAKING
    # ========================================================

    (
        decision,
        audio,
        jalan_masih_lurus,
        person_in_middle
    ) = decision_making(
        band_percentages,
        person_columns
    )


    # ========================================================
    # AUDIO CONTROL
    # ========================================================

    current_time = time.time()


    if (
        decision != last_audio
        and
        current_time - last_audio_time
        >= AUDIO_INTERVAL
    ):

        play_audio(
            audio
        )


        last_audio = decision

        last_audio_time = (
            current_time
        )


    # ========================================================
    # DISPLAY BAND INFORMATION
    # ========================================================

    for i, percentage in enumerate(
        band_percentages
    ):

        cv2.putText(
            result,
            f"B{i + 1}: {percentage:.1f}%",
            (
                10,
                55 + i * 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


    # ========================================================
    # DISPLAY PERSON INFORMATION
    # ========================================================

    cv2.putText(
        result,
        f"Person: {person_count}",
        (
            10,
            FRAME_HEIGHT - 90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"Column: {person_columns}",
        (
            10,
            FRAME_HEIGHT - 60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY FINAL STATUS
    # ========================================================

    if jalan_masih_lurus:

        road_text = (
            "JALAN LURUS: TRUE"
        )

    else:

        road_text = (
            "JALAN LURUS: FALSE"
        )


    if person_in_middle:

        person_text = (
            "ORANG TENGAH: TRUE"
        )

    else:

        person_text = (
            "ORANG TENGAH: FALSE"
        )


    cv2.putText(
        result,
        road_text,
        (
            250,
            FRAME_HEIGHT - 90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        result,
        person_text,
        (
            250,
            FRAME_HEIGHT - 60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )


    # ========================================================
    # DISPLAY DECISION
    # ========================================================

    cv2.putText(
        result,
        decision,
        (
            10,
            FRAME_HEIGHT - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "SIDEWALK EDGE DEVICE",
        result
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    key = cv2.waitKey(1)


    if key == ord("q"):

        break


# ============================================================
# 18. CLEANUP
# ============================================================

print()
print("=" * 60)
print("STOPPING SYSTEM")
print("=" * 60)


cap.release()

cv2.destroyAllWindows()

pygame.mixer.quit()


print(
    "System stopped."
)
