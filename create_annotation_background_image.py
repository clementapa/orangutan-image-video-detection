import glob
import os
import os.path as osp
import shutil

from sklearn.model_selection import train_test_split

root_images = "datasets/10_monkey_species_kaggle"
# root_images = "datasets/Monkey_Species_Data"

list_images = []
for extension in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
    list_images.extend(
        glob.glob(osp.join(root_images, "**", extension), recursive=True)
    )
list_images.sort()

labels = [path.split("/")[-2] for path in list_images]

path_image_train, path_image_val = train_test_split(
    list_images, test_size=0.2, stratify=labels
)

save_directory = root_images.split("/")[-1] + "_annotation"
os.makedirs(save_directory, exist_ok=True)

annotation_directory = osp.join(save_directory, "labels")
image_directory = osp.join(save_directory, "images")
os.makedirs(annotation_directory, exist_ok=True)
os.makedirs(image_directory, exist_ok=True)

for path_img in list_images:
    f_path_short = osp.basename(path_img)
    with open(
        osp.join(annotation_directory, f_path_short.split(".")[0] + ".txt"), "w"
    ) as f:
        pass
    shutil.copy(path_img, osp.join(image_directory, f_path_short))

with open(osp.join(save_directory, "train.txt"), "w") as fp:
    fp.write(
        "\n".join(
            osp.join(image_directory, osp.basename(path)) for path in path_image_train
        )
    )

with open(osp.join(save_directory, "validation.txt"), "w") as fp:
    fp.write(
        "\n".join(
            osp.join(image_directory, osp.basename(path)) for path in path_image_val
        )
    )
