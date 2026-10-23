
"""
AirMouse Pro — Five-Finger Recognition
======================================

Real-time hand tracking and finger-name detection
using OpenCV and MediaPipe Tasks API.

Developer: MD. Atiqul Islam (Atik)
Email: atik.cmttiu1001@gmail.com

Python: 3.10+
Controls:
    Q / ESC : Quit
"""

from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


# ============================================================
# PROJECT INFORMATION
# ============================================================

APP_NAME = "AirMouse Pro"
DEVELOPER = "MD. Atiqul Islam (Atik)"
EMAIL = "atik.cmttiu1001@gmail.com"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# OpenCV uses BGR color values.
BG_COLOR = (25, 20, 38)
WHITE = (245, 245, 255)
CYAN = (255, 220, 40)
GREEN = (75, 255, 120)
RED = (80, 80, 255)
PURPLE = (220, 100, 255)
GRAY = (155, 155, 175)
ORANGE = (30, 175, 255)

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGER_TIPS = {
    "Thumb": THUMB_TIP,
    "Index": INDEX_TIP,
    "Middle": MIDDLE_TIP,
    "Ring": RING_TIP,
    "Pinky": PINKY_TIP,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def download_model() -> None:
    """Download the hand landmark model if it is missing."""

    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000:
        print("Hand landmark model found.")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".part")

    print("Downloading MediaPipe hand landmark model...")

    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "AirMouse-Pro/1.0"},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if temporary_path.stat().st_size < 100_000:
            raise RuntimeError("The downloaded model is incomplete.")

        temporary_path.replace(MODEL_PATH)
        print("Model downloaded successfully.")

    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand model: {exc}\n"
            f"Download it manually from:\n{MODEL_URL}\n"
            f"Save it to:\n{MODEL_PATH}"
        ) from exc


def create_landmarker():
    """Create a MediaPipe video-mode hand landmarker."""

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# FINGER DETECTION
# ============================================================

def joint_angle(a, b, c) -> float:
    """Return the angle ABC in degrees."""

    vector_ba = (a.x - b.x, a.y - b.y)
    vector_bc = (c.x - b.x, c.y - b.y)

    magnitude = (
        math.hypot(*vector_ba)
        * math.hypot(*vector_bc)
    )

    if magnitude < 1e-8:
        return 0.0

    cosine = (
        vector_ba[0] * vector_bc[0]
        + vector_ba[1] * vector_bc[1]
    ) / magnitude

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def distance(a, b) -> float:
    """Calculate normalized landmark distance."""

    return math.hypot(a.x - b.x, a.y - b.y)


def finger_is_extended(hand, mcp, pip, dip, tip) -> bool:
    """
    Estimate extension using finger joint angles.

    The y-coordinate check is intentionally avoided for the four
    fingers because the hand can rotate in the camera image.
    """

    pip_angle = joint_angle(
        hand[mcp], hand[pip], hand[dip]
    )

    dip_angle = joint_angle(
        hand[pip], hand[dip], hand[tip]
    )

    return pip_angle > 150 and dip_angle > 135


def detect_fingers(hand) -> dict[str, bool]:
    """
    Return the estimated extension state of all five fingers.

    True means extended; False means curled or not confidently
    classified as extended.
    """

    hand_size = max(
        distance(hand[WRIST], hand[9]),
        1e-6,
    )

    # Estimate thumb extension using distance from the palm.
    thumb_extended = (
        distance(hand[THUMB_TIP], hand[5])
        > distance(hand[2], hand[5]) + 0.15 * hand_size
    )

    return {
        "Thumb": thumb_extended,

        "Index": finger_is_extended(
            hand, 5, 6, 7, 8
        ),

        "Middle": finger_is_extended(
            hand, 9, 10, 11, 12
        ),

        "Ring": finger_is_extended(
            hand, 13, 14, 15, 16
        ),

        "Pinky": finger_is_extended(
            hand, 17, 18, 19, 20
        ),
    }


# ============================================================
# DRAWING HELPERS
# ============================================================

def put_text(
    frame,
    text,
    position,
    color=WHITE,
    scale=0.5,
    thickness=1,
):
    """Draw anti-aliased text."""

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_hand(frame, hand, finger_states):
    """Draw the skeleton, landmarks, and finger-name labels."""

    height, width = frame.shape[:2]

    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in hand
    ]

    # Draw connections first so landmarks remain visible.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            CYAN,
            2,
            cv2.LINE_AA,
        )

    # Highlight finger tips.
    for index, point in enumerate(points):
        is_tip = index in FINGER_TIPS.values()

        if is_tip:
            cv2.circle(
                frame,
                point,
                10,
                GREEN,
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                point,
                4,
                WHITE,
                -1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(
                frame,
                point,
                3,
                PURPLE,
                -1,
                cv2.LINE_AA,
            )

    # Place a name beside each fingertip.
    for name, tip_index in FINGER_TIPS.items():
        x, y = points[tip_index]

        extended = finger_states[name]
        color = GREEN if extended else RED

        label_x = max(5, min(width - 100, x + 13))
        label_y = max(20, min(height - 12, y - 12))

        put_text(
            frame,
            name,
            (label_x, label_y),
            color,
            0.48,
            2,
        )


def draw_dashboard(
    frame,
    finger_states,
    fps,
    hand_detected,
):
    """Draw the application header, finger panel, and footer."""

    height, width = frame.shape[:2]

    # Header panel.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 125),
        BG_COLOR,
        -1,
    )

    # Footer panel.
    cv2.rectangle(
        overlay,
        (0, height - 115),
        (width, height),
        BG_COLOR,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame,
    )

    # Application title.
    put_text(
        frame,
        APP_NAME,
        (18, 32),
        CYAN,
        0.85,
        2,
    )

    put_text(
        frame,
        "INTELLIGENT HAND TRACKING SYSTEM",
        (20, 55),
        WHITE,
        0.43,
        1,
    )

    # Live FPS.
    put_text(
        frame,
        f"FPS: {fps:.1f}",
        (width - 135, 33),
        GREEN,
        0.60,
        2,
    )

    # Tracking status.
    status = (
        "HAND TRACKING ACTIVE"
        if hand_detected
        else "SHOW YOUR HAND TO CAMERA"
    )

    status_color = GREEN if hand_detected else ORANGE

    put_text(
        frame,
        status,
        (20, 86),
        status_color,
        0.48,
        2,
    )

    # Finger states panel.
    put_text(
        frame,
        "FINGER DETECTION",
        (20, height - 91),
        CYAN,
        0.50,
        2,
    )

    column_width = max(90, (width - 40) // 5)

    for i, (name, extended) in enumerate(finger_states.items()):
        x = 20 + i * column_width
        y = height - 57

        color = GREEN if extended else RED
        state = "OPEN" if extended else "CLOSED"

        cv2.circle(
            frame,
            (x + 5, y - 5),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )

        put_text(
            frame,
            name,
            (x + 16, y),
            WHITE,
            0.43,
            1,
        )

        put_text(
            frame,
            state,
            (x + 16, y + 19),
            color,
            0.38,
            1,
        )

    # Developer credits.
    put_text(
        frame,
        f"Developer: {DEVELOPER}",
        (20, height - 12),
        WHITE,
        0.36,
        1,
    )

    # Put email in the same footer, on the right.
    email_scale = 0.36

    put_text(
        frame,
        f"Email: {EMAIL}",
        (width - 340, height - 12),
        CYAN,
        email_scale,
        1,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    camera = None
    landmarker = None

    try:
        print("=" * 55)
        print(f"{APP_NAME} - Five-Finger Recognition")
        print(f"Developer: {DEVELOPER}")
        print(f"Email: {EMAIL}")
        print("=" * 55)

        download_model()
        landmarker = create_landmarker()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Cannot open camera. Check webcam permissions, "
                "close other camera apps, or try camera index 1."
            )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        start_time = time.monotonic()
        previous_frame_time = start_time
        last_timestamp_ms = -1

        fps = 0.0

        print("\nCamera started successfully.")
        print("Show your hand in front of the webcam.")
        print("Press Q or ESC to exit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Warning: Failed to read camera frame.")
                break

            # Mirror the webcam image.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1,
            )

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            hand_detected = bool(result.hand_landmarks)

            finger_states = {
                name: False for name in FINGER_TIPS
            }

            if hand_detected:
                hand = result.hand_landmarks[0]

                finger_states = detect_fingers(hand)

                draw_hand(
                    frame,
                    hand,
                    finger_states,
                )

            # Smoothed FPS estimate.
            now = time.monotonic()
            delta = now - previous_frame_time

            if delta > 0:
                current_fps = 1.0 / delta

                fps = (
                    current_fps
                    if fps == 0
                    else 0.85 * fps + 0.15 * current_fps
                )

            previous_frame_time = now

            draw_dashboard(
                frame,
                finger_states,
                fps,
                hand_detected,
            )

            cv2.imshow(
                APP_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nApplication stopped by user.")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        print(
            "\nCheck your Python environment, dependencies, "
            "internet connection, and webcam."
        )

    finally:
        if camera is not None:
            camera.release()

        if landmarker is not None:
            landmarker.close()

        cv2.destroyAllWindows()

        print(f"\n{APP_NAME} closed.")
        print(f"Developer: {DEVELOPER}")
        print(f"Email: {EMAIL}")


if __name__ == "__main__":
    main()
