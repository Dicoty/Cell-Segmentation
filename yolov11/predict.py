from ultralytics import YOLO
from PIL import Image
import numpy as np
import imgviz
from torch.nn import functional as F

def np2mask(tif_np):
    """
    二维np转彩色mask,返回的mask为rgba格式
    """
    alpha = np.where(tif_np > 0, 255, 0).astype(np.uint8) # 创建alpha通道
    npmask = imgviz.label2rgb(tif_np)
    rgba = np.dstack([npmask, alpha])
    mask_rgba = Image.fromarray(rgba) # only uint8 can do

    return mask_rgba

def scale_masks(masks, shape, padding: bool = True):
    """
    Rescale segment masks to target shape.

    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple): Target height and width as (height, width).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    # 检查原始维度信息，三维就再扩展为四维
    original_dims = len(masks.shape)
    if original_dims == 3:
        masks = masks.unsqueeze(1)


    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))) if padding else (0, 0)  # y, x
    bottom, right = (
        mh - int(round(pad[1] + 0.1)),
        mw - int(round(pad[0] + 0.1)),
    )
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="nearest")  # NCHW

    # 如何原始图像是三维，结果也恢复为三维
    if original_dims == 3:
        masks = masks.squeeze(1)

    return masks

def yolo_predict(img_path, model_path):

    model = YOLO(model_path)
    results = model.predict(img_path) # results是一个列表
    cell_result = results[0]

    original_shape = cell_result.orig_shape
    original_shape_for_im = (original_shape[1], original_shape[0]) # Iamge的宽高顺序和返回的相反，要调过来

    mask_each_cell = scale_masks(cell_result.masks.data, original_shape) # (细胞数， 宽， 高)，即每个细胞一层，这里的宽高为640*448，和使用的模型有关

    mask_each_cell_np = mask_each_cell.data.cpu().numpy() # 装换为纯numpy数组
    
    # 每个细胞乘以权重以转换为类似于tif格式
    n = mask_each_cell_np.shape[0]
    indices = np.arange(1, n+1).reshape(n, 1, 1)
    weighted_mask_each = indices * mask_each_cell_np

    tif_mask_float = weighted_mask_each.max(axis=0) # 装换为类似tif格式
    tif_mask = tif_mask_float.astype(np.uint8)

    # # 扩展为原始大小
    # im_tif_mask = Image.fromarray(tif_mask)
    # tif_mask_resized = im_tif_mask.resize(original_shape_for_im, Image.NEAREST)

    # 重回numpy数组
    tif_mask_resized_np = np.array(tif_mask)

    return tif_mask_resized_np

if __name__ == "__main__":

    image = r"D:\20251012\20251012-fixedA375-30ugPaint_JL_2\1.tif"
    model = r"C:\somefiles\weights_and_dataset\yolo_weights\train\weights\best.pt"

    tif_mask = yolo_predict(image, model)
    np.save(r"D:\20251012\JL2_mask", tif_mask)
    mask_rgba = np2mask(tif_mask)
    mask_rgba.show()