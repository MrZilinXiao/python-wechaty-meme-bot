import logging
import sys
import yaml


def Singleton(cls):
    """
    A decorator for Singleton support
    """
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("wechaty-meme-bot")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)


@Singleton
class ConfigParser:
    config_dict = None

    def __init__(self, config_path='config.yaml'):  # inited
        with open(config_path, 'r') as buf:
            ConfigParser.config_dict = yaml.load(buf, Loader=yaml.FullLoader)

    @staticmethod
    def get_dict():
        if ConfigParser.config_dict is not None:
            return ConfigParser.config_dict
        else:
            raise RuntimeError('Config parser not inited...')
