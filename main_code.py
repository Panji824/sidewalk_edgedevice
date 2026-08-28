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
# 2. KONFIGURASI SISTEM
# ============================================================

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_PATH = "./model_tobias/sidewalk_tobias.onnx"


# ------------------------------------------------------------
# Kamera
# ------------------------------------------------------------

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ------------------------------------------------------------
# Ukuran input model
# ------------------------------------------------------------

MODEL_WIDTH = 224
MODEL_HEIGHT = 224


# ------------------------------------------------------------
# Class hasil segmentasi SegFormer
# ------------------------------------------------------------

SIDEWALK_CLASS = 2
PERSON_CLASS = 8


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

NUM_BANDS = 6
NUM_COLUMNS = 5


# ------------------------------------------------------------
# Threshold keputusan B4
#
# B4 > 8%  = jalan masih lurus
# B4 <= 8% = sidewalk berakhir
# ------------------------------------------------------------

B4_THRESHOLD = 8.0


# ------------------------------------------------------------
# Minimum luas objek manusia
# ------------------------------------------------------------

MIN_PERSON_AREA = 100


# ------------------------------------------------------------
# Interval audio
# ------------------------------------------------------------

AUDIO_INTERVAL = 3.0


# ============================================================
# 3. KONFIGURASI AUDIO
# ============================================================

AUDIO_DIR = "./audio"


# ------------------------------------------------------------
# Audio yang tersedia pada repository
# ------------------------------------------------------------

AUDIO_ADA_ORANG = os.path.join(
    AUDIO_DIR,
    "ada_orang.mp3"
)

AUDIO_BELOK_KANAN = os.path.join(
    AUDIO_DIR,
    "belok kanan.mp3"
)

AUDIO_BELOK_KIRI = os.path.join(
    AUDIO_DIR,
    "belok kiri.mp3"
)

AUDIO_BERHENTI = os.path.join(
    AUDIO_DIR,
    "berhenti.mp3"
)

AUDIO_LURUS = os.path.join(
    AUDIO_DIR,
    "lurus.mp3"
)

AUDIO_PUTAR_BALIK = os.path.join(
    AUDIO_DIR,
    "putar balik.mp3"
)


# ============================================================
# 4. INISIALISASI AUDIO
# ============================================================

print("=" * 60)
print("INITIALIZING AUDIO")
print("=" * 60)

pygame.mixer.init()

print("[AUDIO] Audio system initialized.")


# ------------------------------------------------------------
# Cek file audio
# ------------------------------------------------------------

audio_files = [
    AUDIO_ADA_ORANG,
    AUDIO_BELOK_KANAN,
    AUDIO_BELOK_KIRI,
    AUDIO_BERHENTI,
    AUDIO_LURUS,
    AUDIO_PUTAR_BALIK
]


for audio_file in audio_files:

    if os.path.exists(audio_file):

        print(
            "[AUDIO] OK :",
            audio_file
        )

    else:

        print(
            "[AUDIO] MISSING :",
            audio_file
        )


# ============================================================
# 5. FUNGSI MEMUTAR AUDIO
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
            "[AUDIO ERROR]:",
            e
        )


# ============================================================
# 6. LOAD MODEL ONNX
# ============================================================

print()
print("=" * 60)
print("LOADING ONNX MODEL")
print("=" * 60)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )


# ------------------------------------------------------------
# ONNX Runtime menggunakan CPU Raspberry Pi
# ------------------------------------------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)


# ============================================================
# 7. INFORMASI MODEL
# ============================================================

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
# 8. PREPROCESSING FRAME
# ============================================================

def preprocess(frame):

    """
    Preprocessing frame kamera sebelum
    masuk ke model SegFormer.
    """

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            MODEL_WIDTH,
            MODEL_HEIGHT
        ),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # Convert ke float32
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalisasi 0-1
    # --------------------------------------------------------

    image = image / 255.0


    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (
            2,
            0,
            1
        )
    )


    # --------------------------------------------------------
    # Tambahkan batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    return image.astype(
        np.float32
    )


# ============================================================
# 9. INFERENSI SEGMENTASI
# ============================================================

def run_segmentation(frame):

    """
    Menjalankan model SegFormer ONNX
    untuk mendapatkan hasil segmentasi.
    """

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    input_tensor = preprocess(
        frame
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME:
            input_tensor
        }
    )


    logits = outputs[0]


    # --------------------------------------------------------
    # Ambil class dengan nilai terbesar
    # --------------------------------------------------------

    prediction = np.argmax(
        logits,
        axis=1
    )[0]


    # --------------------------------------------------------
    # Resize hasil segmentasi
    # ke ukuran frame kamera
    # --------------------------------------------------------

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
# 10. MEMBUAT MASK SIDEWALK
# ============================================================

def create_sidewalk_mask(
    prediction
):

    """
    Mengambil class sidewalk
    dari hasil segmentasi.
    """

    sidewalk_mask = (
        prediction ==
        SIDEWALK_CLASS
    ).astype(
        np.uint8
    )


    return sidewalk_mask


# ============================================================
# 11. MEMBUAT MASK PERSON
# ============================================================

def create_person_mask(
    prediction
):

    """
    Mengambil class person
    dari hasil segmentasi.
    """

    person_mask = (
        prediction ==
        PERSON_CLASS
    ).astype(
        np.uint8
    )


    return person_mask


# ============================================================
# 12. MENGHITUNG PERSENTASE SIDEWALK PER BAND
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

        # ----------------------------------------------------
        # Batas band
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Ambil area band
        # ----------------------------------------------------

        band = sidewalk_mask[
            y_start:y_end,
            :
        ]


        # ----------------------------------------------------
        # Hitung pixel sidewalk
        # ----------------------------------------------------

        sidewalk_pixels = np.sum(
            band == 1
        )


        # ----------------------------------------------------
        # Total pixel
        # ----------------------------------------------------

        total_pixels = band.size


        # ----------------------------------------------------
        # Persentase sidewalk
        # ----------------------------------------------------

        percentage = (
            sidewalk_pixels /
            total_pixels
        ) * 100


        percentages.append(
            percentage
        )


    return percentages


# ============================================================
# 13. MENGGAMBAR GRID 6 BAND
# ============================================================

def draw_bands(
    frame
):

    """
    Membuat 6 pembagian horizontal:
    B1 sampai B6.
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


    # --------------------------------------------------------
    # Label band
    # --------------------------------------------------------

    for i in range(
        NUM_BANDS
    ):

        y = int(
            i *
            band_height
        )


        cv2.putText(
            frame,
            f"B{i + 1}",
            (
                10,
                y + 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


    return frame


# ============================================================
# 14. MENGGAMBAR GRID 5 KOLOM
# ============================================================

def draw_columns(
    frame
):

    """
    Membagi gambar menjadi 5 kolom:

    C1 = kiri
    C2 = kiri tengah
    C3 = tengah
    C4 = kanan tengah
    C5 = kanan
    """

    height, width = (
        frame.shape[:2]
    )


    column_width = (
        width /
        NUM_COLUMNS
    )


    # --------------------------------------------------------
    # Garis vertikal
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Label kolom
    # --------------------------------------------------------

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
            2,
            cv2.LINE_AA
        )


    return frame


# ============================================================
# 15. DETEKSI PERSON DAN CENTROID
# ============================================================

def detect_person(
    person_mask,
    frame
):

    """
    Mendeteksi area manusia dari person mask.

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


    # --------------------------------------------------------
    # Connected Components
    # --------------------------------------------------------

    (
        num_labels,
        labels,
        stats,
        centroids
    ) = cv2.connectedComponentsWithStats(
        person_mask,
        connectivity=8
    )


    person_count = 0

    person_columns = []


    # --------------------------------------------------------
    # Proses setiap objek
    # --------------------------------------------------------

    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]


        # ----------------------------------------------------
        # Abaikan objek terlalu kecil
        # ----------------------------------------------------

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
        # Bounding box
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
        # Gambar centroid
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
        # Label person
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
            2,
            cv2.LINE_AA
        )


    return (
        person_count,
        person_columns
    )


# ============================================================
# 16. VISUALISASI HASIL SEGMENTASI
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


    # --------------------------------------------------------
    # Sidewalk
    # --------------------------------------------------------

    overlay[
        sidewalk_mask == 1
    ] = (
        0,
        255,
        0
    )


    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    overlay[
        person_mask == 1
    ] = (
        0,
        0,
        255
    )


    # --------------------------------------------------------
    # Gabungkan dengan frame asli
    # --------------------------------------------------------

    result = cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.3,
        0
    )


    return result


# ============================================================
# 17. DECISION MAKING
# ============================================================

def decision_making(
    band_percentages,
    person_columns
):

    """
    Decision making sistem.

    Aturan utama:

    1. Person berada di C3
       -> Ada orang di tengah
       -> Audio ada_orang.mp3

    2. Tidak ada person di C3 dan B4 > 8%
       -> Jalan masih lurus
       -> Audio lurus.mp3

    3. Tidak ada person di C3 dan B4 <= 8%
       -> Sidewalk berakhir
       -> Audio berhenti.mp3
    """


    # --------------------------------------------------------
    # Ambil B4
    # --------------------------------------------------------

    B4 = band_percentages[3]


    # --------------------------------------------------------
    # Cek person di tengah
    # C3 = kolom nomor 3
    # --------------------------------------------------------

    person_in_middle = (
        3 in person_columns
    )


    # --------------------------------------------------------
    # Kondisi jalan
    # --------------------------------------------------------

    jalan_masih_lurus = (
        B4 > B4_THRESHOLD
    )


    # ========================================================
    # PRIORITAS 1
    # ADA ORANG DI TENGAH
    # ========================================================

    if person_in_middle:

        decision = (
            "ADA ORANG DI TENGAH"
        )

        audio = (
            AUDIO_ADA_ORANG
        )


    # ========================================================
    # PRIORITAS 2
    # JALAN MASIH LURUS
    # ========================================================

    elif jalan_masih_lurus:

        decision = (
            "JALAN MASIH LURUS"
        )

        audio = (
            AUDIO_LURUS
        )


    # ========================================================
    # PRIORITAS 3
    # SIDEWALK BERAKHIR
    # ========================================================

    else:

        decision = (
            "SIDEWALK BERAKHIR"
        )

        audio = (
            AUDIO_BERHENTI
        )


    return (
        decision,
        audio,
        jalan_masih_lurus,
        person_in_middle
    )


# ============================================================
# 18. INISIALISASI CAMERA
# ============================================================

print()
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
    "[CAMERA] Camera started."
)


# ============================================================
# 19. MAIN STREAMING LOOP
# ============================================================

print()
print("=" * 60)
print("SIDEWALK SYSTEM RUNNING")
print("=" * 60)

print(
    "Tekan Q untuk keluar."
)


# ------------------------------------------------------------
# Variabel audio
# ------------------------------------------------------------

last_decision = None

last_audio_time = 0


# ============================================================
# LOOP UTAMA
# ============================================================

while True:

    # ========================================================
    # 19.1 CAPTURE FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "[CAMERA] Gagal membaca frame."
        )

        break


    # ========================================================
    # 19.2 SEGMENTATION
    # ========================================================

    prediction = run_segmentation(
        frame
    )


    # ========================================================
    # 19.3 SIDEWALK MASK
    # ========================================================

    sidewalk_mask = (
        prediction ==
        SIDEWALK_CLASS
    ).astype(
        np.uint8
    )


    # ========================================================
    # 19.4 PERSON MASK
    # ========================================================

    person_mask = (
        prediction ==
        PERSON_CLASS
    ).astype(
        np.uint8
    )


    # ========================================================
    # 19.5 VISUALISASI SEGMENTASI
    # ========================================================

    result = draw_segmentation(
        frame,
        sidewalk_mask,
        person_mask
    )


    # ========================================================
    # 19.6 HITUNG BAND
    # ========================================================

    band_percentages = (
        calculate_bands(
            sidewalk_mask
        )
    )


    # ========================================================
    # 19.7 DETEKSI PERSON
    # ========================================================

    (
        person_count,
        person_columns
    ) = detect_person(
        person_mask,
        result
    )


    # ========================================================
    # 19.8 GRID BAND
    # ========================================================

    result = draw_bands(
        result
    )


    # ========================================================
    # 19.9 GRID COLUMN
    # ========================================================

    result = draw_columns(
        result
    )


    # ========================================================
    # 19.10 DECISION MAKING
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
    # 19.11 AUDIO CONTROL
    # ========================================================

    current_time = time.time()


    # Audio dimainkan ketika decision berubah
    # dan sudah melewati interval minimum.

    if (
        decision != last_decision
        and
        (
            current_time -
            last_audio_time
        ) >= AUDIO_INTERVAL
    ):

        play_audio(
            audio
        )


        last_decision = decision

        last_audio_time = (
            current_time
        )


    # ========================================================
    # 19.12 TAMPILKAN PERSENTASE BAND
    # ========================================================

    for i, percentage in enumerate(
        band_percentages
    ):

        cv2.putText(
            result,
            f"B{i + 1}: {percentage:.1f}%",
            (
                10,
                55 + i * 24
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # 19.13 TAMPILKAN PERSON
    # ========================================================

    cv2.putText(
        result,
        f"Person: {person_count}",
        (
            10,
            FRAME_HEIGHT - 90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Kolom person
    # --------------------------------------------------------

    cv2.putText(
        result,
        f"Column: {person_columns}",
        (
            10,
            FRAME_HEIGHT - 60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 19.14 STATUS JALAN
    # ========================================================

    if jalan_masih_lurus:

        road_text = (
            "JALAN MASIH LURUS: TRUE"
        )

    else:

        road_text = (
            "JALAN MASIH LURUS: FALSE"
        )


    cv2.putText(
        result,
        road_text,
        (
            250,
            FRAME_HEIGHT - 90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 19.15 STATUS ORANG TENGAH
    # ========================================================

    if person_in_middle:

        person_text = (
            "ADA ORANG DI TENGAH: TRUE"
        )

    else:

        person_text = (
            "ADA ORANG DI TENGAH: FALSE"
        )


    cv2.putText(
        result,
        person_text,
        (
            250,
            FRAME_HEIGHT - 60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 19.16 TAMPILKAN DECISION
    # ========================================================

    cv2.putText(
        result,
        decision,
        (
            10,
            FRAME_HEIGHT - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 19.17 TAMPILKAN B4
    # ========================================================

    cv2.putText(
        result,
        f"B4 THRESHOLD: {B4_THRESHOLD:.0f}%",
        (
            400,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 19.18 DISPLAY
    # ========================================================

    cv2.imshow(
        "SIDEWALK EDGE DEVICE",
        result
    )


    # ========================================================
    # 19.19 KEYBOARD CONTROL
    # ========================================================

    key = cv2.waitKey(1)


    if key == ord("q"):

        break


# ============================================================
# 20. CLEANUP
# ============================================================

print()
print("=" * 60)
print("STOPPING SYSTEM")
print("=" * 60)


cap.release()

cv2.destroyAllWindows()

pygame.mixer.quit()


print(
    "[SYSTEM] System stopped."
)
