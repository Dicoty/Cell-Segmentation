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
    只保留img中有mask的区域,其他地方置为纯白,传入的均为Image对象
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

def calculate(img, mask):
    """
    传入图片(Image对象)和mask(tifffile读取的tif,即np数组),返回mask下面积在原图中的占比,检测出的细胞数,mask下的G-R总值和平均值,信号热图(Image对象)
    """
    mask_np = np.array(mask)
    cell_nums = len(np.unique(mask_np)) - 1
    image_np = np.array(img)
    mask_np_bool = mask_np > 0
    cell_R = image_np[:, :, 0] * mask_np_bool
    cell_G = image_np[:, :, 1] * mask_np_bool
    sum_GR_sub = np.sum(cell_G - cell_R)
    cell_area = np.sum(mask_np_bool)
    all_area = image_np.shape[0] * image_np.shape[1]
    ratio = cell_area / all_area
    GR_mean = sum_GR_sub / cell_area

    GR_sub = cell_G - cell_R

    # 使用imgviz或matplotlib生成热图
    import matplotlib.pyplot as plt
    import io
    
    # 创建热图
    plt.figure(figsize=(image_np.shape[1]/100, image_np.shape[0]/100), dpi=100)
    plt.imshow(GR_sub, cmap='jet')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # 保存到内存中的buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    
    # 转换为PIL Image
    from PIL import Image
    heatmap_img = Image.open(buf)
    
    # 关闭图形以释放内存
    plt.close()

    return ratio, cell_nums, sum_GR_sub, GR_mean, heatmap_img
