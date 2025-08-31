# 将labelme标注的实例分割数据集转化为yolo格式，注意yolo的目标检测和实例分割格式差异大，本工具只能转化实例分割
# 注意，由于写入坐标是用的是a模式，所以每次对同一labelme标注数据集运行本工具时，请将上一次生成的数据集删除，重新生成
import json
import os
import shutil
import random
import yaml
random.seed(42)  # 设置随机种子以确保结果可复现

# labelme标注文件目录
labelme_dir = r"C:\somefiles\weights_and_dataset\Segmentation_Annotations"

# 验证集占总数据集的比例
val_ratio = 0.2

assert os.path.exists(labelme_dir), "Labelme annotation directory does not exist."
parent_dir = os.path.dirname(labelme_dir)
# 创建yolo数据集的标注文件夹格式
yolo_dir = os.path.join(parent_dir, "yolo_anno")# 此文件夹下放yaml
if not os.path.exists(yolo_dir):
    os.makedirs(yolo_dir)
yolo_dataset_dir = os.path.join(yolo_dir, "yolo_dataset")
if not os.path.exists(yolo_dataset_dir):
    os.makedirs(yolo_dataset_dir)
image_dir = os.path.join(yolo_dataset_dir, "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
image_train_dir = os.path.join(image_dir, "train")
if not os.path.exists(image_train_dir):
    os.makedirs(image_train_dir)
image_val_dir = os.path.join(image_dir, "val")
if not os.path.exists(image_val_dir):
    os.makedirs(image_val_dir)
label_dir = os.path.join(yolo_dataset_dir, "labels")
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
label_train_dir = os.path.join(label_dir, "train")
if not os.path.exists(label_train_dir):
    os.makedirs(label_train_dir)
label_val_dir = os.path.join(label_dir, "val")
if not os.path.exists(label_val_dir):
    os.makedirs(label_val_dir)

all_anno = [f for f in os.listdir(labelme_dir) if f.endswith(".json")]
all_anno.sort()
all_classes = []

val = random.sample(all_anno, int(len(all_anno) * val_ratio))

for anno in all_anno:
    anno_path = os.path.join(labelme_dir, anno)

    with open(anno_path, "r") as f:
        json_anno = json.load(f)
    height, width = json_anno["imageHeight"], json_anno["imageWidth"]
    image_name = json_anno["imagePath"]
    # 将图片文件放入对应文件夹
    labelme_image_path = os.path.join(labelme_dir, image_name)
    if anno in val:
        shutil.copy(labelme_image_path, os.path.join(image_val_dir, image_name))
    else:
        shutil.copy(labelme_image_path, os.path.join(image_train_dir, image_name))
    # 对标注的形状进行转化
    for shape in json_anno["shapes"]:
        # 将不同形状存入列表
        shape_type = shape['label']
        if shape_type not in all_classes:
            all_classes.append(shape_type)
        # 点坐标归一化装换
        points = shape['points']
        point_list = []
        for point in points:
            x = point[0] / width
            y = point[1] / height
            point_list.append(x)
            point_list.append(y)
        # 写入txt文件
        if anno in val:
            label_file_path = os.path.join(label_val_dir, os.path.splitext(image_name)[0] + ".txt")
        else:
            label_file_path = os.path.join(label_train_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(label_file_path, "a") as label_file:
            label_file.write(str(all_classes.index(shape_type)) + " " + " ".join([str(x) for x in point_list]) + "\n")

all_classes_dict = {i: name for i, name in enumerate(all_classes)}
# 创建yaml文件
yaml_data = {  # yaml数据
    "path": os.path.basename(yolo_dataset_dir),
    "train": os.path.relpath(image_train_dir, start=yolo_dataset_dir),
    "val": os.path.relpath(image_val_dir, start=yolo_dataset_dir),
    "names": all_classes_dict,
}
with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
    yaml.dump(yaml_data, f)