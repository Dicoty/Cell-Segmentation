import numpy as np
from cellpose import plot, utils, io
from matplotlib import pyplot as plt
import os
import pandas as pd
# 要运行还要下载openpyxl库

# 图片和npy标注文件的文件夹，手动输入
file_folder = r"C:\somefiles\cellimage\train"
# 图片格式，手动输入
img_extension = ".jpg"

all_anno_name = [os.path.join(file_folder, f) for f in os.listdir(file_folder) if f.endswith('.npy')]
all_img_name = [f.replace("_seg.npy", img_extension) for f in all_anno_name]

for f in all_img_name:
    assert os.path.exists(f), "{} not exist!".format(f)

result_folder = os.path.join(file_folder, "result")
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

excel_file_path = os.path.join(result_folder, "result.xlsx")

mask_img_folder = os.path.join(result_folder, "mask")
if not os.path.exists(mask_img_folder):
    os.mkdir(mask_img_folder)

data = {}
img_names = []
cell_num = []
cell_in_view_ratio = []
cell_sum_GR = []
cell_mean_GR = []

for anno in all_anno_name:
    # dat是一个字典
    dat = np.load(anno, allow_pickle=True).item()

    img = anno.replace("_seg.npy", img_extension)
    img_name = os.path.basename(img).split(".")[0]
    img_names.append(img_name)
    # 返回[h,w,3], RGB
    img = io.imread(img)

    # plot image with masks overlaid
    mask_RGB = plot.mask_overlay(img, dat['masks'],
                            colors=np.array(dat['colors']))
    plt.imshow(mask_RGB)
    mask_img_name = "mask_" + img_name + ".png"
    mask_img_path = os.path.join(mask_img_folder, mask_img_name)
    plt.savefig(mask_img_path)


    cell_num.append(len(dat['colors']))

    masks = dat['masks']
    heigth, weight = masks.shape

    cell_in_view_ratio.append(np.sum(masks > 0) / (heigth * weight)*100)

    bool_mask = masks
    bool_mask[bool_mask > 1] = 1

    # 扩展 bool_mask 的维度以匹配 img 的形状
    bool_mask_3d = np.expand_dims(bool_mask, axis=-1)  # 形状变为 (2048, 3072, 1)
    bool_mask_3d = np.repeat(bool_mask_3d, 3, axis=-1)  # 形状变为 (2048, 3072, 3)

    math_img = img * bool_mask_3d # RGB

    G = math_img[1]
    R = math_img[0]

    sub = G - R

    cell_sum_GR.append(sub.sum())
    cell_mean_GR.append(sub.mean())


data["图片名"] = img_names
data["细胞数"] = cell_num
data["细胞在视野中的占比"] = cell_in_view_ratio
data["G-R总和"] = cell_sum_GR
data["G-R均值"] = cell_mean_GR

df = pd.DataFrame(data)
df.to_excel(excel_file_path, index=False)