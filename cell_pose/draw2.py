import numpy as np
from cellpose import plot, utils, io
from matplotlib import pyplot as plt

"""
    这里的代码可以用来画标注的蒙版
"""

dat = np.load(r"C:\somefiles\cellimage\train\2_seg.npy", allow_pickle=True).item()
img = io.imread(r"C:\somefiles\cellimage\train\2.jpg")

# plot image with masks overlaid
mask_RGB = plot.mask_overlay(img, dat['masks'],
                        colors=np.array(dat['colors']))

plt.imshow(mask_RGB)
plt.show()


# plot image with outlines overlaid in red
outlines = utils.outlines_list(dat['masks'])
plt.imshow(img)
for o in outlines:
    plt.plot(o[:,0], o[:,1], color='r')

plt.show()
