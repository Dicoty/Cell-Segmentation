from cellpose import models, io
import numpy as np
from pathlib import Path

def cp_predict(img, model_path, flow_threshold=0.4, cellprob_threshold=0.0, tile_norm_blocksize=0):
    img_path = Path(img)
    model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    image = io.imread(img_path)

    first_channel = '0' # @param ['None', 0, 1, 2, 3, 4, 5]
    second_channel = '1' # @param ['None', 0, 1, 2, 3, 4, 5]
    third_channel = '2' # @param ['None', 0, 1, 2, 3, 4, 5]

    selected_channels = []
    for i, c in enumerate([first_channel, second_channel, third_channel]):
        if c == 'None':
            continue
        if int(c) > image.shape[-1]:
            assert False, 'invalid channel index, must have index greater or equal to the number of channels'
        if c != 'None':
            selected_channels.append(int(c))
    
    img_selected_channels = np.zeros_like(img)
    img_selected_channels[:, :, :len(selected_channels)] = img[:, :, selected_channels]

    masks, flows, styles = model.eval(img_selected_channels, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                      normalize={"tile_norm_blocksize": tile_norm_blocksize})
    
    return masks

img_path = r'C:\somefiles\Cell-Segmentation\2.jpg'
model_path = r"C:\somefiles\cellimage\models\cpsam_20250713_201512"
mask = cp_predict(img_path, model_path)

if mask is not None:
    print(mask.shape)
    print('预测完成')
else:
    print('预测失败')