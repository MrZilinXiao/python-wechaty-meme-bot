from requests.adapters import HTTPAdapter
from wechaty_puppet import FileBox, ScanStatus  # type: ignore
from wechaty_puppet import MessageType

from wechaty import Wechaty, Contact
from wechaty.user import Message, Room

import requests
from requests.packages.urllib3.util import Retry
from PIL import Image
import os
from frontend import config


class MemeBot(Wechaty):
    def __init__(self):
        super(MemeBot, self).__init__()
        if not os.path.exists(config.image_temp_dir):
            os.mkdir(config.image_temp_dir)

    def _img_to_PIL(self, img):
        pass

    # def

    async def on_message(self, msg: Message):
        from_contact = msg.talker()
        img_path = None
        if msg.is_self():  # for self testing
            if msg.type() == MessageType.MESSAGE_TYPE_IMAGE:
                img = await msg.to_file_box()
                data_param = {
                    'img_name': img.name,
                    'data': img.base64
                }
                s = requests.Session()
                s.mount('http://',
                        HTTPAdapter(max_retries=Retry(total=3, method_whitelist=frozenset(['GET', 'POST']))))  # allow retry when encountering connection issue
                response = s.post(url=config.backend_url, data=data_param)

                # img_path = os.path.join(config.image_temp_dir, img.name)
                # await img.to_file(img_path)


