from backend.ocr_wrapper import OCRWrapper
from orm import Meme
import difflib
from fuzzywuzzy import fuzz
from PIL import Image
import numpy as np
import random


class RequestDispatcher(object):
    """
    A dispatcher which receives a image and takes corresponding action according to image content:
    1. No OCR Results -> backend.response.feature
    2. OCR Results exist and match one of database entries -> return it
    3. OCR Results exist but no match with database entries -> 1)
    """

    def __init__(self, ocr_backbone='mobilenetv2'):
        self.OCR = OCRWrapper(ocr_backbone)
        self.meme_list = []  # [path, title, [tag1, tag2, ...]]
        self._read_in_db()

    def _read_in_db(self):
        """
        Read in path, title and tags in list for _get_close_matches to find matches.
        :return:
        """
        for meme in Meme.select():
            self.meme_list.append([meme.path, meme.title, [tag for tag in meme.tag.split(' ')]])

    def receive_handler(self, img_path: str):
        """
        After web handler receives a image, its path gets passed here, hoping to get a meme response.
        A `History` entry will be submitted into database for record.

        :param img_path: received image path sent by web_handler
        :return: str, response meme image path
        """
        receive_img = Image.open(img_path).convert("RGB")
        receive_img = np.array(receive_img)
        text_list = self.OCR.text_predict(receive_img)
        random.shuffle(self.meme_list)  # shuffle meme_list before each iteration to
        # avoid the situation where strategy always answers with the same image
        for word in text_list:  # for each word in received meme image
            img_path = self._matched(word)
            if img_path is not None:
                return img_path
        return self.meme_list[0][0]  # if with no luck, return a random meme image
        # TODO: should be dispatcher to backend.response.feature

    def _matched(self, target: str):
        """
        Determine whether there is a match in title, tags
        :param target:
        :return: bool
        """
        for path, title, tags in self.meme_list:
            if RequestDispatcher._get_close_matches(target, title) or RequestDispatcher._get_close_matches(target,
                                                                                                           tags):
                return path
        return None

    @staticmethod
    def _get_close_matches(target: str, src, top_similarity: int = 1, cutoff: float = 0.6):
        """
        Wrapper of difflib for close matches
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
            return fuzz.ratio(target, src) > cutoff
