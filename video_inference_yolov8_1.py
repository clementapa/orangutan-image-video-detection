import os.path as osp

import numpy as np
import supervision as sv
from supervision import Color
from ultralytics import YOLO

YOLO_MODEL = YOLO("orang_outan_detection/train7/weights/best.pt")
BOX_ANNOTATOR = sv.BoxAnnotator(color=Color.from_hex("#FF00E4"))

path_video = "VID20230502094937.mp4"
path_video = "stock-footage-female-sumatran-orangutan-and-baby-sitting-on-tree-branch-against-green-foliage-on-background.webm"
path_video = "stock-footage-pair-of-northwest-bornean-orangutans-on-the-ground-larger-one-covered-in-dusty-wood-chip-shavings.webm"
video_info = sv.VideoInfo.from_video_path(path_video)


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    output = YOLO_MODEL(frame, imgsz=640, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(output)

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


sv.process_video(
    source_path=path_video,
    target_path=osp.basename(path_video).split(".")[0] + "_detection.mp4",
    callback=process_frame,
)
