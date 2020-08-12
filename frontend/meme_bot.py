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
import hashlib
import uuid


class MemeBot(Wechaty):
    def __init__(self):
        super(MemeBot, self).__init__()
        self.cache_dict = {}
        if not os.path.exists(config.image_temp_dir):
            os.mkdir(config.image_temp_dir)
        else:
            self._load_cache()

    def _load_cache(self):
        for img_file in os.walk(config.image_temp_dir):  # type: str
            if img_file.lower().endswith(config.allow_img_extensions):
                self.cache_dict[hashlib.md5(open(img_file, 'rb').read()).hexdigest()] = img_file

    def _img_to_PIL(self, img):
        pass

    async def on_message(self, msg: Message):
        from_contact = msg.talker()
        if msg.is_self():  # for self testing
            if msg.type() == MessageType.MESSAGE_TYPE_IMAGE:
                img = await msg.to_file_box()
                data_param = {
                    'img_name': img.name,
                    'data': img.base64
                }
                s = requests.Session()
                s.mount('http://',
                        HTTPAdapter(max_retries=Retry(total=3, method_whitelist=frozenset(
                            ['GET', 'POST']))))  # allow retry when encountering connection issue
                ret_json = s.post(url=config.backend_upload_url, data=data_param).json()   # keys: img_name, md5
                # example returning json: {'img_name': '/001/001.jpg', 'md5': 'ff7bd2b664bf65962a924912bfd17507'}
                if ret_json['md5'] in self.cache_dict:  # hit cache
                    ret_path = self.cache_dict[ret_json['md5']]
                else:
                    ret_img = s.get(url=config.backend_static_url + ret_json['img_name'])
                    ret_path = os.path.join(config.image_temp_dir, str(uuid.uuid4()) + '.' +
                                            ret_json['img_name'].split('.')[-1])
                    with open(ret_path, 'wb') as f:
                        f.write(ret_img.content)

                ret_img = FileBox.from_file(ret_path, name=os.path.basename(ret_path))
                await msg.say(ret_img)