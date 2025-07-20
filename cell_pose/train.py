import numpy as np
from cellpose import models, core, io, plot, train, metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import torch

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
    raise ImportError("No GPU access, change your runtime")

def main():
    # CellposeModel 的可调参数
    pretrained_model = "cpsam" # 默认是“cpsam”
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = models.CellposeModel(gpu=True, 
                                 pretrained_model=pretrained_model,
                                 device=device)

    # 训练数据所在的文件夹
    train_dir = r"C:\somefiles\weights_and_dataset\cellpose_anno\cell"
    if not Path(train_dir).exists():
        raise FileNotFoundError("directory does not exist")

    test_dir = None # optionally you can specify a directory with test files

    # *** change to your mask extension // 文件扩展名 ***
    masks_ext = "_masks"
    # ^ assumes images from Cellpose GUI, if labels are tiffs, then "_masks.tif"

    # list all files
    files = [f for f in Path(train_dir).glob("*") if "_masks" not in f.name and "_flows" not in f.name and "_seg" not in f.name]

    if(len(files)==0):
        raise FileNotFoundError("no files found, did you specify the correct folder and extension?")
    else:
        print(f"{len(files)} files in folder:")

    # 打印文件名
    for f in files:
      print(f.name)

    model_name = "new_model"

    # default training params
    n_epochs = 30
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1
    save_every = 10

    # get files
    output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
    train_data, train_labels, _, test_data, test_labels, _ = output
    # (not passing test data into function to speed up training)

    new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                                train_data=train_data,
                                                                train_labels=train_labels,
                                                                batch_size=batch_size,
                                                                n_epochs=n_epochs,
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                nimg_per_epoch=max(2, len(train_data)), # can change this
                                                                model_name=model_name,
                                                                save_every=save_every,
                                                                save_each=True) # save_each不设为True就没办法save_every
    
    print(new_model_path)
    parent_path = os.path.dirname(new_model_path)

    loss_file = os.path.join(parent_path, f"{model_name}_loss.json")
    print(loss_file)
    loss_data = {"train_loss": train_losses, "val_loss": test_losses}
    with open(loss_file, "w") as f:
        json.dump(loss_data, f)
    print(f"Loss history saved to {loss_file}")

if __name__ == '__main__':
    main()