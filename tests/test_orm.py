import pytest
from orm import Meme


def test_read_database():
    from backend.response.dispatcher import BaseHandler
    handler = BaseHandler()
    handler._read_in_db()
    assert isinstance(handler.meme_list, list)
    assert len(handler.meme_list) > 0


if __name__ == '__main__':
    pytest.main()
