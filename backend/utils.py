import logging
import sys


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("Fighting-Meme")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("logs/main.log")
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
