"""
ORM Abstraction Module for peewee
"""
from peewee import *

db = SqliteDatabase('test.db')
db.connect()


class BaseModel(Model):
    """
    BaseModel for peewee
    """

    class Meta:
        database = db


class MemeType(BaseModel):
    """
    Model for meme type
    """
    title = CharField(verbose_name='类标题', max_length=256)
    centroid = TextField(verbose_name='分类质心，以`,`分割')
    is_clustered = BooleanField(verbose_name='是否是自行聚类的结果', default=False)


class Meme(BaseModel):
    """
    Model for single meme image
    """
    path = CharField(verbose_name='表情相对路径', max_length=256, unique=True)
    title = CharField(verbose_name='表情标题', max_length=256)
    tag = CharField(verbose_name='表情标签，以`,`分割', max_length=512, default='')
    raw_text = TextField(verbose_name='原始OCR结果')
    feature = TextField(verbose_name='表情特征（numpy array序列化后）')
    clusterType = ForeignKeyField(MemeType, backref='Meme', on_delete='NO ACTION', null=True)


class History(BaseModel):
    """
    Model for logging meme response history
    """
    receive_img_path = CharField(verbose_name='收到表情路径', max_length=256)
    receive_img_feature = TextField(verbose_name='收到表情特征', default='')
    response_log = TextField(verbose_name='回复逻辑')
    response_img_path = CharField(verbose_name='回复表情路径', max_length=256)


if __name__ == '__main__':
    # executed only for once to build sqlite database
    try:
        db.create_tables([Meme, MemeType, History], safe=True)
        db.close()
        print("Create Tables Successfully!")
    except IntegrityError as e:
        print("Init Database Error with Exception {}".format(str(e)))
        raise e
    except Exception as e:
        print("Init Database Failed with Unknown Error: {}".format(str(e)))
        raise e