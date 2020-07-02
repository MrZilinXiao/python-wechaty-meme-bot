from backend.chineseocr_lite.model import crnnRec
from backend.chineseocr_lite.psenet import PSENet, PSENetHandel
from backend.chineseocr_lite.config import *
import numpy as np


class OCRWrapper(object):
    def __init__(self, pse_model_type='mobilenetv2'):
        if pse_model_type == "mobilenetv2":
            text_detect_net = PSENet(backbone=pse_model_type, pretrained=False, result_num=6, scale=pse_scale)
        self.text_handle = PSENetHandel(pse_model_path, text_detect_net, pse_scale, gpu_id=GPU_ID)

    def text_predict(self, img) -> list:
        preds, boxes_list, rects_re, t = self.text_handle.predict(img, long_size=pse_long_size)
        result = crnnRec(np.array(img), rects_re)
        return [text['text'] for text in result]
