import hanlp

from backend.utils import ConfigParser


class HanlpWrapper(object):
    def __init__(self):
        self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        self.stopDict = {}
        # StopWords List comes from: https://github.com/goto456/stopwords
        # Read stopwords into dict to ensure constant reference consumption
        with open(ConfigParser.config_dict['general']['stop_words_path'], 'r') as f:
            for word in f.readlines():
                self.stopDict[word.strip()] = True

    def is_stop_word(self, word: str) -> bool:
        return word in self.stopDict

    def Tokenizer(self, text: str) -> list:
        raw_list = self.tokenizer(text)
        raw_list = list(set(raw_list))  # remove repetition
        return [word for word in raw_list if not self.is_stop_word(word)]
