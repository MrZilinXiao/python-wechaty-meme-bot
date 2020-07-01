"""
表情包预处理，仅需在表情包数据集变动时调用
"""
import os
from PIL import Image


def png2jpg(png_path: str):
    # TODO: png->jpg job should be done in reading phrase instead of preprocessing since raw gif file is needed.
    im = Image.open(png_path)
    rgb_im = im.convert("RGB")
    rgb_im.save(png_path[:-4] + '.jpg')
    os.remove(png_path)


def gif2jpg(gif_path: str):  # The same using Pillow if only used to extract the first frame
    png2jpg(gif_path)


def preprocess(img_path: str):
    for root, dirs, files in os.walk(img_path, topdown=True):
        if not dirs:  # for each subdir
            for name in files:  # for each meme
                if name.endswith('.png'):
                    png2jpg(os.path.join(root, name))
                elif name.endswith('.gif'):
                    gif2jpg(os.path.join(root, name))
