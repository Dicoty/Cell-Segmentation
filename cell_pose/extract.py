import numpy as np
from PIL import Image
import os

# 存放原图和预测图的文件夹
# 成功运行会在同目录下创建一个叫output的文件夹，内有处理后的图片
path = r""

parent_dir = os.path.dirname(path)
output_dir = os.path.join(parent_dir, "output")

print(f"输出文件夹为: {output_dir}")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def extract_image(img_path, mask_path):
    """
    将img中的细胞提取出来,其他地方置为白色,RGB三通道
    """
    Imask = Image.open(mask_path)
    mask = np.array(Imask)
    Iimg = Image.open(img_path)
    img = np.array(Iimg)

    rgb_img = np.ones_like(img) * 255 # 空白背景
    mask_condition = mask > 0
    for i in range(img.shape[2]):  # 对每个颜色通道
        rgb_img[:, :, i] = np.where(mask_condition, img[:, :, i], rgb_img[:, :, i])

    return rgb_img

for filename in os.listdir(path):
    if filename.endswith(".png"):
        mask_path = os.path.join(path, filename)
        img_path = os.path.join(path, filename.split("_")[0] + ".jpg")
        if os.path.exists(img_path):
            exeract_img = extract_image(img_path, mask_path)
            Iexeract_img = Image.fromarray(exeract_img)
            Iexeract_img.save(os.path.join(output_dir, filename.split(".")[0] + "_extract.png"))
            print(f"{filename} done")
        else:
            print(f"找到{mask_path}, 但未找到{img_path}")