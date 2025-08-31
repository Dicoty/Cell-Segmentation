from cellpose import models
from cellpose.io import imread
from cellpose import metrics, io
import torch
from pathlib import Path

# CellposeModel 的可调参数
pretrained_model = "cpsam" # 默认是“cpsam”
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = models.CellposeModel(gpu=True, 
                            pretrained_model=pretrained_model,
                            device=device)

# 训练数据所在的文件夹
train_dir = None
if not Path(train_dir).exists():
    raise FileNotFoundError("directory does not exist")

test_dir = ""

# *** change to your mask extension // 文件扩展名 ***
masks_ext = "_masks"
# ^ assumes images from Cellpose GUI, if labels are tiffs, then "_masks"

# get files
output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
train_data, train_labels, _, test_data, test_labels, _ = output

# 模型预测时的可调参数
diameter = 30 # 设置为30*N, 则表示下采样N倍
batch_size = 8 # 最小为1，一般设置为8的倍数
flow_threshold = 0.4 # 调整范围为[0.1,3.0]
cellprob_threshold = 0.0 # 调整范围为[-6,6]

# run model on test images
masks = model.eval(test_data, 
                   batch_size=batch_size)[0]

# check performance using ground truth labels
ap = metrics.average_precision(test_labels, masks)[0]
print('')
print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')