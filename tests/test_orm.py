import pytest
from orm import Meme


def test_read_database():
    from backend.response.dispatcher import BaseHandler
    handler = BaseHandler()
    handler._read_in_db()
    assert isinstance(handler.meme_list, list)
    assert len(handler.meme_list) > 0


def test_insert_and_remove_database():
    path = './test_path'
    title = 'test title'
    tag = 'test tag'
    feature = 'test_feature'
    Meme.create(path=path, title=title, tag=tag, feature=feature)
    test_record = Meme.get(Meme.path == path)
    assert test_record.delete_instance() == 1


if __name__ == '__main__':
    pytest.main()
