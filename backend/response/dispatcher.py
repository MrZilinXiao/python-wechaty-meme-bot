from backend.ocr_wrapper import OCRWrapper
from orm import Meme
from typing import List
from PIL import Image
import numpy as np
import random
import os


class BaseHandler(object):
    """
    Base class for handling separated strings
    ConversationHandler & DirectHandler all derive from this class
    Those classes will share `meme_list` static var.
    """
    meme_list = None  # static variable shared by all Handler

    def __init__(self):
        if self.meme_list is None:
            self.meme_list = []
            self._read_in_db()

    def get_matched(self, target: List[str]) -> (str, List[str]):
        """
        :param target: List of OCR strings,
        :return: (str, List[str]) -> response meme path, log_list
        """
        pass

    def _read_in_db(self):
        """
        Read in path, title and tags in list for _get_close_matches to find matches.
        :return:
        """
        for meme in Meme.select():
            self.meme_list.append([meme.path, meme.title, [tag for tag in meme.tag.split(' ')]])


class RequestDispatcher(object):
    """
    A dispatcher which receives a image and takes the corresponding action according to image content:
    1. No OCR Results -> random response
    2. OCR Results exist:
        2.1 put tokenized result words into single-conversation model -> search model response in database
            2.1.1 if hit: return hit path
            2.1.2 if not hit: go to 2.2
        2.2 search tokenized result words directly in database, return answer if hit, go to 2.3 if not
        2.3 (test idea) put model response onto preset meme images.

    """

    def __init__(self, ocr_backbone='mobilenetv2'):
        self.OCR = OCRWrapper(ocr_backbone)
        self.meme_list = []  # [path, title, [tag1, tag2, ...]]

    def receive_handler(self, img_path: str) -> (str, list):
        """
        After web handler receives a image, its path gets passed here, hoping to get a meme response.
        A `History` entry will be submitted into database for record.

        :param img_path: received image path sent by web_handler
        :return: (str, log_list)
        """
        log_list = ['DEBUG记录: ']  # list to log response strategy
        receive_img = Image.open(img_path).convert("RGB")
        receive_img = np.array(receive_img)
        text_list = self.OCR.text_predict(receive_img)
        if text_list:  # there are OCR result(s)
            log_list.append('有OCR结果，结果为{}'.format(' '.join(text_list)))
            random.shuffle(
                self.meme_list)  # shuffle meme_list before each request to avoid the situation where strategy always answers with the same image
            for word in text_list:  # for each word in received meme image
                log_list.append('正在查找表情中"{}"一词是否匹配数据库...'.format(word))
                img_path = self._matched(word)
                if img_path is not None:
                    log_list.append('"{}"成功匹配数据库表情: {}'.format(word, img_path))
                    if not os.path.exists(img_path):
                        log_list.append('文件不存在: {}'.format(img_path))
                        return '', log_list
                    return img_path, log_list
                else:
                    log_list.append('"{}"匹配无果...'.format(word))
            log_list.append('所有词均匹配无果，随机返回一项表情...')
            return self.meme_list[0][0], log_list  # if with no luck, return a random meme image
        else:
            pass  # TODO: should be dispatched to backend.response.feature

