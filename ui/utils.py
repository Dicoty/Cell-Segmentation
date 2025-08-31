import imgviz
import tifffile
from PIL import Image
import numpy as np

def np2mask(tif_np):
    """
    二维np转彩色mask,返回的mask为rgba格式
    """
    alpha = np.where(tif_np > 0, 255, 0).astype(np.uint8) # 创建alpha通道
    npmask = imgviz.label2rgb(tif_np)
    rgba = np.dstack([npmask, alpha])
    mask_rgba = Image.fromarray(rgba) # only uint8 can do

    return mask_rgba

def sub_img(img, mask):
    """
    只保留img中有mask的区域，其他地方置为纯白
    """
    # mask为rgba
    mask_alpha = mask.split()[-1]
    alpha_array = np.array(mask_alpha)

    img_array = np.array(img)
    if len(img_array.shape) == 3:  # 彩色图像
        # 创建一个白色背景的图像
        rgb_image = np.ones_like(img_array, dtype=np.uint8) * 255
        # 将mask区域内的像素复制到新图像中
        mask_condition = alpha_array > 0
        for i in range(img_array.shape[2]):  # 对每个颜色通道
            rgb_image[:, :, i] = np.where(mask_condition, img_array[:, :, i], rgb_image[:, :, i])
    else:  # 灰度图像
        # 创建一个白色背景的图像
        rgb_image = np.ones_like(img_array, dtype=np.uint8) * 255
        # 将mask区域内的像素复制到新图像中
        rgb_image = np.where(alpha_array > 0, img_array, rgb_image)
        
        # 如果需要转换为RGB彩色图像，可以将灰度图像复制到三个通道
        rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=2)

    # 转换为PIL图像（RGB模式）
    result_img = Image.fromarray(rgb_image, mode='RGB')

    return result_img