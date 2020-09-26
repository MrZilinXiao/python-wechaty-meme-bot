from backend.response.conversation import ConversationHandler
from backend.response.dispatcher import DirectHandler
from backend.meme_importer import BaseImporter
import pytest


def test_import():
    # disable GPU-NICE
    importer = BaseImporter(meme_path='./backend/meme/', num_workers=0, extractor_type='none', max_count=1)
    importer.import_meme()


def test_conversation_matched(conversation_handler: ConversationHandler):
    ret_meme, log = conversation_handler.get_matched(['你好呀'])
    assert isinstance(ret_meme, str)
    assert isinstance(log, list)
    print(ret_meme)
    print(log)


def test_close_match(direct_handler: DirectHandler):
    assert direct_handler._get_close_matches('hello', 'Hello')
    assert direct_handler._get_close_matches('hello', ['Hello', 'hallo', 'hi'])


if __name__ == '__main__':
    pytest.main()
