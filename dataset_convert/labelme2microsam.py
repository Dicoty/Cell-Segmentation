# 将labelme标注的分割文件转化为micro_sam可用的tif格式的标注，可用于训练micro_sam
# 对于实例分割，每一类放置于一个文件夹内
import os
import numpy as np
import labelme
from labelme import utils
import shutil
import json
from PIL import Image

labelme_dir = r"C:\somefiles\weights_and_dataset\Segmentation_Annotations"

cellpose_anno_ext = ".tif"
parent_dir = os.path.dirname(labelme_dir)
all_anno = [f for f in os.listdir(labelme_dir) if f.endswith(".json")]
cellpose_dir = os.path.join(parent_dir, "microsam_anno")
if not os.path.exists(cellpose_dir): os.makedirs(cellpose_dir)

for anno in all_anno:
    anno_path = os.path.join(labelme_dir, anno)

    label_file = labelme.LabelFile(filename=anno_path)
    with open(anno_path, "r") as f:
        json_anno = json.load(f)
    image_shape = [json_anno["imageHeight"], json_anno["imageWidth"], 3]
    image_name = json_anno["imagePath"]
    labelme_image_path = os.path.join(labelme_dir, image_name)

    # 由于cellpose只能对一个类别做分割，不同类别的数据不能放在一起
    label_name_to_value = {"_background_": 0}
    for shape in sorted(label_file.shapes, key=lambda x: x["label"]): # 按照标签名称排序
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    # lbl是一个二维数组，大小和被标注图片相同，不同类别的对象mask对应位置的值分别标为1，2，3……
    # ins与lbl形状相同，但是对每一个实例mask都用不同的值标记
    cla, ins = utils.shapes_to_label(image_shape, label_file.shapes, label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    
    # label_names是一个列表，0位是背景，其他位与label_name_to_value对应
    for name, value in label_name_to_value.items():
        label_names[value] = name  # type: ignore[call-overload]

    # 创建文件路径
    class_dir = []
    for name in label_names:
        class_dir.append(os.path.join(cellpose_dir, name))

    for i in range(1, len(class_dir)):
        # 检查文件夹是否存在，不存在则创建，背景不创建
        if not os.path.exists(class_dir[i]):
            os.makedirs(class_dir[i])

        temp_cla = cla.copy()
        temp_ins = ins.copy()
        temp_cla[temp_cla != i] = 0 # 将非当前类别的像素值设为0
        temp_cla[temp_cla > 0] = 1 # 将当前类别的像素值设为1,类似bool

        temp_ins = temp_ins * temp_cla
        image_name_without_ext = os.path.splitext(image_name)[0]

        if not os.path.exists(os.path.join(class_dir[i], 'labels')):
            os.makedirs(os.path.join(class_dir[i], 'labels'))
        if not os.path.exists(os.path.join(class_dir[i], 'images')):
            os.makedirs(os.path.join(class_dir[i], 'images'))

        label_path = os.path.join(class_dir[i], 'labels', image_name_without_ext + cellpose_anno_ext)
        # 保存tif格式的标注文件
        label = Image.fromarray(temp_ins)
        label.save(label_path)
        # 保存图片,micro_sam只支持tif格式的灰度图
        gray_img = Image.open(labelme_image_path).convert('L')  # 转换为灰度图
        gray_img_path = os.path.join(class_dir[i], 'images', image_name_without_ext + ".tif")
        gray_img.save(gray_img_path)
