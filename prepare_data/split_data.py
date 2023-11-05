import glob
import os
import os.path as osp

import yaml
from sklearn.model_selection import train_test_split

path_dataset = "../datasets/dataset_orang_outan_annotation"

list_images = []
for extension in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
    list_images.extend(
        glob.glob(osp.join(path_dataset, "images", "**", extension), recursive=True)
    )
list_images.sort()

path_image_train, path_image_val = train_test_split(list_images, test_size=0.2)

split_directory = osp.join(path_dataset, "split")
os.makedirs(split_directory, exist_ok=True)

image_directory = osp.join(path_dataset.split("/")[-1], "images")

with open(osp.join(split_directory, "train.txt"), "w") as fp:
    fp.write(
        "\n".join(
            osp.join(image_directory, osp.basename(path)) for path in path_image_train
        )
    )

with open(osp.join(split_directory, "validation.txt"), "w") as fp:
    fp.write(
        "\n".join(
            osp.join(image_directory, osp.basename(path)) for path in path_image_val
        )
    )

yaml_file = {
    "train": "train.txt",
    "val": "validation.txt",
    "nc": 1,
    "names": ["orang_outan"],
}
with open(osp.join(split_directory, "data.yaml"), "w") as outfile:
    yaml.dump(yaml_file, outfile, default_flow_style=False)
