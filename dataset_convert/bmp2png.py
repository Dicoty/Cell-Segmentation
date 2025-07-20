from PIL import Image
import os

# 指定文件夹路径
input_folder = r"C:\somefiles\weights_and_dataset\cellpose_anno\cell"
output_folder = r"C:\somefiles\weights_and_dataset\cellpose_anno\cell"

# 如果输出文件夹不存在，创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的 .bmp 文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".bmp"):
        # 构建完整文件路径
        bmp_path = os.path.join(input_folder, filename)
        # 打开图像
        img = Image.open(bmp_path)
        # 构建对应的 .png 文件名
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(output_folder, png_filename)
        # 保存为 .png 格式
        img.save(png_path, "PNG")
        print(f"Converted: {bmp_path} -> {png_path}")

# 可选：删除原始 .bmp 文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".bmp"):
        os.remove(os.path.join(input_folder, filename))
        print(f"Deleted: {filename}")