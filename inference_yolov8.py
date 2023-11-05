import cv2
import supervision as sv
from ultralytics import YOLO

YOLO_MODEL = YOLO("orang_outan_detection/train7/weights/best.pt")
BOX_ANNOTATOR = sv.BoxAnnotator()

path_image = "orang-outan-arbre.jpg"

output = YOLO_MODEL(path_image, verbose=False)[0]

detections = sv.Detections.from_ultralytics(output)

labels = [
    f"{output.names[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _ in detections
]

annotated_frame = BOX_ANNOTATOR.annotate(
    scene=output.orig_img.copy(), detections=detections, labels=labels
)

cv2.imwrite("annotate_image.jpg", annotated_frame)
