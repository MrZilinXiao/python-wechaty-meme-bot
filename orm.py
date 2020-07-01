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
    tag = CharField(verbose_name='类标签，以`,`分割', max_length=512, default='')
    centroid = TextField(verbose_name='分类质心，以`,`分割')


class Meme(BaseModel):
    path = CharField(verbose_name='表情相对路径', max_length=256, unique=True)
    title = CharField(verbose_name='表情标题', max_length=256)
    tag = CharField(verbose_name='表情标签，以`,`分割', max_length=512, default='')
    clusterType = ForeignKeyField(MemeType, backref='Meme')
