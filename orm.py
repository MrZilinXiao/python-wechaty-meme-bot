"""
ORM Abstraction Module for peewee

"""
from peewee import *

db = SqliteDatabase('test.db')


class BaseModel(Model):
    class Meta:
        database = db


class MemeType(BaseModel):
    title = CharField(verbose_name='类标题', max_length=256)
    centroid = TextField(verbose_name='分类质心，以`,`分割')
    is_clustered = BooleanField(verbose_name='是否是自行聚类的结果', default=False)


class Meme(BaseModel):
    path = CharField(verbose_name='表情相对路径', max_length=256, unique=True)
    title = CharField(verbose_name='表情标题', max_length=256)
    tag = CharField(verbose_name='表情标签，以`,`分割', max_length=512, default='')
    feature = TextField(verbose_name='表情特征（numpy array序列化后）')
    clusterType = ForeignKeyField(MemeType, backref='Meme')
