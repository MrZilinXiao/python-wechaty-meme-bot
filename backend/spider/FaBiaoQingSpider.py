"""
This is a template of BaseSpider usage.
"""
from bs4 import BeautifulSoup

from backend.spider.BaseSpider import BaseSpider
from queue import Queue
import requests
import os
from typing import Union


class FaBiaoQingSpider(BaseSpider):
    def __init__(self, queue: Union[Queue, None], download_path: str):
        super(FaBiaoQingSpider, self).__init__(queue=queue, download_path=download_path)

    def download(self, meme_url: str, title=None):
        response = requests.get(meme_url)
        soup = BeautifulSoup(response.content, 'lxml')
        img_list = soup.find_all('img', class_='ui image lazy')

        for img in img_list:
            image = img.get('data-original')
            title = img.get('title')
            print('Downloading image: ', title)
            try:
                with open(self.download_path + title + os.path.splitext(image)[-1], 'wb') as f:
                    img = requests.get(image).content
                    f.write(img)
            except OSError:
                print('save failed')
                break


if __name__ == '__main__':
    _url = 'https://fabiaoqing.com/biaoqing/lists/page/{}.html'
    queue = Queue()
    spider = FaBiaoQingSpider(queue, './backend/meme/others/')  # master thread
    spider.add_url([_url.format(pg) for pg in range(1, 201)])
    for _ in range(10):  # 10 slave threads
        worker = FaBiaoQingSpider(None, './backend/meme/others/')
        worker.daemon = True
        worker.start()
    queue.join()  # until queue is empty
