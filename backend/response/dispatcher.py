from backend.ocr_wrapper import OCRWrapper
import difflib
import os
from typing import Union, List
from fuzzywuzzy import fuzz
from orm import Meme
from PIL import Image
import numpy as np
import random


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


class DirectHandler(BaseHandler):
    """
    A class derived from BaseHandler, with extra ability to search keywords in local database
    """

    def __init__(self):
        super(DirectHandler, self).__init__()

    def get_matched(self, target: List[str], log=None) -> (str, List[str]):
        log_list = [] if not isinstance(log, list) else log
        img_path = None
        for kw in target:   # for each keyword in target list
            img_path = self._matched(kw)
            if img_path is None:
                log_list.append('"{}"匹配无果...'.format(kw))
            else:
                log_list.append('"{}"成功匹配数据库表情: {}'.format(kw, img_path))
                if not os.path.exists(img_path):
                    log_list.append('文件不存在: {}'.format(img_path))
                    img_path = None
                break
        if img_path is None:
            log_list.append('所有词均匹配无果，随机返回一项表情...')
            random.shuffle(self.meme_list)
            return self.meme_list[0][0], log_list
        return img_path, log_list

    def _matched(self, target: str) -> Union[str, None]:
        """
        Determine whether there is a match in title, tags
        :param target
        :return: str/None
        """
        for path, title, tags in self.meme_list:
            if self._get_close_matches(target, title) or self._get_close_matches(target, tags):
                return path
        return None

    @staticmethod
    def _get_close_matches(target: str, src: Union[List[str], str], top_similarity: int = 1, cutoff: float = 0.5):
        """
        Wrapper of difflib/fuzzywuzzy for close matches
        :param target: target string you want to match
        :param src: list/str
        if src is list, return False if there is no match in the entire src list, vice versa.
        if src is str, return False if similar ratio is lower than `cutoff`
        :param top_similarity:
        :param cutoff:
        :return: bool, indicating if there is a match for target in src
        """
        if isinstance(src, list):
            return bool(difflib.get_close_matches(target, src, n=top_similarity, cutoff=cutoff))
        elif isinstance(src, str):
            return fuzz.ratio(target, src) > cutoff * 100


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
        from backend.response.conversation import ConversationHandler
        self.OCR = OCRWrapper(ocr_backbone)
        # self.meme_list = []  # [path, title, [tag1, tag2, ...]]
        self.conversation = ConversationHandler()

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
            log_list.append('有OCR结果，结果为“{}”'.format(' '.join(text_list)))
            # random.shuffle(
            #    self.meme_list)  # shuffle meme_list before each request to avoid the situation where strategy always answers with the same image
            return self.conversation.get_matched(text_list, log_list)
        else:
            pass  # TODO: should be dispatched to backend.response.feature

