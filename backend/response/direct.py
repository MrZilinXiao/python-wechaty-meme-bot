import difflib
import os
from typing import Union, List
from fuzzywuzzy import fuzz
from backend.response.dispatcher import BaseHandler


class DirectHandler(BaseHandler):
    """
    A class derived from BaseHandler, with extra ability to search keywords in local database
    """

    def __init__(self):
        super(DirectHandler, self).__init__()

    def get_matched(self, target: List[str], log=None) -> (str, List[str]):
        log_list = [] if not isinstance(log, list) else log
        for kw in target:   # for each keyword in target list
            img_path = self._matched(kw)
            if img_path is None:
                log_list.append('"{}"匹配无果...'.format(kw))
            else:
                log_list.append('"{}"成功匹配数据库表情: {}'.format(kw, img_path))
                if not os.path.exists(img_path):
                    log_list.append('文件不存在: {}'.format(img_path))
                    img_path = ''
                break
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
    def _get_close_matches(target: str, src: Union[List[str], str], top_similarity: int = 1, cutoff: float = 0.6):
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
