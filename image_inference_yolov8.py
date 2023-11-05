import cv2
import supervision as sv
from supervision import Color
from ultralytics import YOLO

YOLO_MODEL = YOLO("orang_outan_detection/train7/weights/best.pt")
BOX_ANNOTATOR = sv.BoxAnnotator(color=Color.from_hex("#FF00E4"))

# path_image = "petit-ouran-outang-mort-pairi-daiza-belgique-960x640.jpg"
path_image = "weekend-chimpanze.jpg"

confidence = 0.6

output = YOLO_MODEL(path_image, verbose=False)[0]

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

cv2.imwrite("annotate_image.jpg", annotated_frame)
