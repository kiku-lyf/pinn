from PIL import Image
import os
from natsort import natsorted

def create_gif(image_folder, output_gif, duration=500):
    # 获取所有图片文件的路径，并按文件名自然排序
    image_files = natsorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('png', 'jpg', 'jpeg', 'bmp'))])

    # 打开图片并存储在一个列表中
    images = [Image.open(image) for image in image_files]

    # 将图片保存为 GIF
    if images:  # 确保图像列表不为空
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

# 调用函数，指定图片文件夹和输出的 GIF 文件名
create_gif('./', 'output.gif', duration=300)
