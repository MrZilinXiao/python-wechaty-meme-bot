import os
from queue import Queue
from threading import Thread
from typing import Union


class BaseSpider(Thread):
    """
    Base spider with multi-thread support
    Downloading from different sites should be implemented via overwriting `download` method
    """
    queue = None

    def __init__(self, queue: Union[Queue, None], download_path: str = './backend/meme/others'):
        Thread.__init__(self)
        self.download_path = download_path
        if BaseSpider.queue is None and queue is not None:
            BaseSpider.queue = queue
        if not os.path.exists(download_path):
            os.mkdir(download_path)

    def add_url(self, url_list: list):
        _ = [self.queue.put(url) for url in url_list]

    def download(self, meme_url: str, title=None):
        return NotImplemented

    def run(self) -> None:
        while True:
            meme_url = self.queue.get()
            try:
                self.download(meme_url)
            finally:
                self.queue.task_done()
