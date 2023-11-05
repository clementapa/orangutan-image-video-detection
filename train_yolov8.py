import os
import os.path as osp

from ultralytics import YOLO

home = os.getcwd()

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

config_path = osp.join(home, "config_train_yolo/default.yaml")

# data_yaml = osp.join(home, "dataset_orang_outan_annotation/split/data.yaml")
data_yaml = "data.yaml"
project = "orang_outan_detection"

model.train(
    data=data_yaml,
    project=project,
    cfg=config_path,
    warmup_epochs=1,
    epochs=100,
    batch=-1,
    workers=4,
)
