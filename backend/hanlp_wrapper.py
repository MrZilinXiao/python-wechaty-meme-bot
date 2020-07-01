import hanlp


class HanlpWrapper(object):
    def __init__(self):
        self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        self.stopDict = {}
        # StopWords List comes from: https://github.com/goto456/stopwords
        # Read stopwords into dict to ensure constant reference consumption
        with open('stopwords.txt', 'r') as f:
            for word in f.readlines():
                self.stopDict[word.strip()] = True

    def IsStopWord(self, word: str) -> bool:
        return word in self.stopDict

    def Tokenizer(self, text: str) -> list:
        rawList = self.tokenizer(text)
        return [word for word in rawList if not self.IsStopWord(word)]
