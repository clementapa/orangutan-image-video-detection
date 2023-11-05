import os.path as osp

import cv2
import numpy as np
import supervision as sv
from supervision import Color
from ultralytics import YOLO

YOLO_MODEL = YOLO("orang_outan_detection/train7/weights/best.pt")
BOX_ANNOTATOR = sv.BoxAnnotator(color=Color.from_hex("#FF00E4"))

path_video = "VID20230502094937.mp4"
path_video = "/home/clement/Documents/personal_project/orang-outan-image-video-detection/hf_space/resources/examples_videos/stock-footage-wild-orangutan-baby-climbing-on-his-own-in-tree-in-bukit-lawang-sumatra-indonesia.mp4"
# path_video = "stock-footage-pair-of-northwest-bornean-orangutans-on-the-ground-larger-one-covered-in-dusty-wood-chip-shavings.webm"
# path_video = "hf_space/resources/examples_videos/VID20230502094937.mp4"
# path_video = "/home/clement/Documents/personal_project/orang-outan-image-video-detection/hf_space/resources/examples_videos/stock-footage-footage-of-big-adult-orang-utan-or-scientifically-known-as-pongo-pygmaeus-walk-on-lean-tree-trunk.mp4"
video_capture = cv2.VideoCapture(path_video)

# Check if the video file was successfully opened
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
frame_rate = int(video_capture.get(5))

output_video = osp.basename(path_video).split(".")[0] + "_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as needed
out = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))


def process_frame(frame: np.ndarray, confidence: float) -> np.ndarray:
    output = YOLO_MODEL(frame, imgsz=640, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(output)

    detections = detections[detections.confidence >= confidence]

    labels = [
        f"{output.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    thickness = 2
    text_thickness = 1
    text_scale = 1.0

    height, width, _ = output.orig_img.shape

    thickness_ratio = ((width + height) / 2) / 400
    text_scale_ratio = ((width + height) / 2) / 600
    text_thickness_ratio = ((width + height) / 2) / 400

    BOX_ANNOTATOR.thickness = int(thickness * thickness_ratio)
    BOX_ANNOTATOR.text_scale = float(text_scale * text_scale_ratio)
    BOX_ANNOTATOR.text_thickness = int(text_thickness * text_thickness_ratio)

    annotated_frame = BOX_ANNOTATOR.annotate(
        scene=output.orig_img.copy(), detections=detections, labels=labels
    )
    return annotated_frame


confidence = 0.3

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Check if the video has ended
    if not ret:
        break

    # Do something with the frame (e.g., display it or process it)
    # For example, you can display the frame in a window
    annotated_frame = process_frame(frame, confidence=confidence)

    out.write(annotated_frame)

# Release the video capture object and close any open windows
video_capture.release()
out.release()
cv2.destroyAllWindows()
