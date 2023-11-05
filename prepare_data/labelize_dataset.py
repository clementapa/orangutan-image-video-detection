import glob
import os
import os.path as osp

import cv2
import supervision as sv
from groundingdino.util.inference import Model
from tqdm import tqdm

HOME = os.getcwd()
CONFIG_PATH = osp.join(
    HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
print(CONFIG_PATH, "; exist:", osp.isfile(CONFIG_PATH))

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = osp.join(HOME, "GroundingDINO.weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", osp.isfile(WEIGHTS_PATH))

model = Model(
    model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH, device="cpu"
)

root_images = "../datasets/dataset_orang_outan/data"
list_images = glob.glob(osp.join(root_images, "**", "*.jpg"), recursive=True)

classes = ["monkey"]
box_threshold = 0.2
text_threshold = 0.2

output_folder = "dataset_detection"

box_annotator = sv.BoxAnnotator()

os.makedirs(output_folder, exist_ok=True)
os.makedirs(osp.join(output_folder, "images_annotated"), exist_ok=True)

progress_bar = tqdm(list_images, desc="Labeling images")
# iterate through images in input_folder
for f_path in progress_bar:
    progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
    image = cv2.imread(f_path)

    f_path_short = osp.basename(f_path)

    images_map = {}
    detections_map = {}

    images_map[f_path_short] = image.copy()
    detections = model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    detections_map[f_path_short] = detections

    dataset = sv.DetectionDataset(["orang_outan"], images_map, detections_map)

    dataset.as_yolo(
        osp.join(output_folder, "images"),
        osp.join(output_folder, "labels"),
        data_yaml_path=osp.join(output_folder, "data.yaml"),
    )

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels
    )

    cv2.imwrite(
        osp.join(output_folder, "images_annotated", f_path_short), annotated_frame
    )
