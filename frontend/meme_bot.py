import base64
import time

from requests.adapters import HTTPAdapter
from wechaty_puppet import FileBox
from wechaty_puppet import MessageType

from wechaty import Wechaty
from wechaty.user import Message

import requests
from requests.packages.urllib3.util import Retry
import os
import hashlib
import uuid
from typing import Dict
import yaml


class MemeBot(Wechaty):
    content_type_mapping = {
        'image/gif': '.gif',
        'image/jpeg': '.jpg',
        'image/png': '.png'
    }

    def __init__(self, debug=False, config='config.yaml'):
        super(MemeBot, self).__init__()
        self.cache_dict = {}
        self.debug = debug
        if os.environ.get('WECHATY_MEME_BOT_CONFIG', None):
            config = os.environ['WECHATY_MEME_BOT_CONFIG']
        self.config_dict: dict = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        if not os.path.exists(self.config_dict['general']['image_temp_dir']):
            os.mkdir(self.config_dict['general']['image_temp_dir'])
        else:
            self._load_cache()  # load meme images received earlier as cache
        self.s = requests.Session()
        self.s.mount('http://',
                     HTTPAdapter(max_retries=Retry(total=3, method_whitelist=frozenset(
                         ['GET', 'POST']))))  # allow retry when encountering connection issue

    def _load_cache(self) -> None:
        for img_file, _, _ in os.walk(self.config_dict['general']['image_temp_dir']):  # type: str
            if img_file.lower().endswith(eval(self.config_dict['general']['allow_img_extensions'])):
                self.cache_dict[hashlib.md5(open(img_file, 'rb').read()).hexdigest()] = img_file

    async def msg_handler(self, msg: Message) -> Dict[str, str]:
        """
        Handling different types of meme, MessageType.MESSAGE_TYPE_IMAGE or MessageType.MESSAGE_TYPE_EMOTICON
        :param msg: Message
        :return: dict like this: {'img_name': '/001/001.jpg', 'md5': 'ff7bd2b664bf65962a924912bfd17507'}
        """
        data_param = {}
        if msg.type() == MessageType.MESSAGE_TYPE_IMAGE:  # message is an image
            img = await msg.to_file_box()
            data_param.update(
                img_name=img.name,
                data=img.base64
            )
        elif msg.type() == MessageType.MESSAGE_TYPE_EMOTICON:  # message is an Wechat EMOTICON, need to fetch from CDN
            import xml.etree.ElementTree as Etree
            content = msg.payload.text  # xml content, need xml parser to extract msg.emoji(cdnurl)
            msgtree = Etree.fromstring(content)
            cdn_url = msgtree.find('emoji').attrib['cdnurl']
            ret = self.s.get(cdn_url)
            b64_str = base64.b64encode(ret.content)
            data_param.update(img_name=str(uuid.uuid4()) + MemeBot.content_type_mapping[ret.headers['Content-Type']],
                              data=b64_str)
        ret_json = self.s.post(url=self.config_dict['backend']['backend_upload_url'], data=data_param).json()  # ret keys: img_name, md5, log
        return ret_json

    async def on_message(self, msg: Message):
        if msg.is_self():  # for self testing
            if msg.type() == MessageType.MESSAGE_TYPE_IMAGE or msg.type() == MessageType.MESSAGE_TYPE_EMOTICON:
                st_time = time.time()
                ret_json = await self.msg_handler(msg)
                # example returning json: {'img_name': '/001/001.jpg', 'md5': 'ff7bd2b664bf65962a924912bfd17507'}
                if ret_json['md5'] in self.cache_dict:  # hit cache
                    ret_path: str = self.cache_dict[ret_json['md5']]
                    if 'log' in ret_json:
                        ret_json['log'] += '\n回复图片命中缓存!'
                else:
                    ret_img = self.s.get(url=self.config_dict['backend']['backend_static_url'] + ret_json['img_name'])
                    if not str(ret_img.status_code).startswith('2'):  # not 2XX response code
                        raise FileNotFoundError(
                            "Can't get img from URL {}, with HTTP status code {}".format(
                                self.config_dict['backend']['backend_static_url'] + ret_json['img_name'], str(ret_img.status_code)))
                    ret_path = os.path.join(self.config_dict['general']['image_temp_dir'], str(uuid.uuid4()) + os.path.extsep +
                                            ret_json['img_name'].split('.')[-1])
                    with open(ret_path, 'wb') as f:
                        f.write(ret_img.content)
                    self.cache_dict[ret_json['md5']] = ret_path
                ret_json['log'] += '\n前后端交互耗时：%.2fs' % (time.time() - st_time)

                if self.debug and 'log' in ret_json:
                    await msg.say(ret_json['log'])

                with open(ret_path, 'rb') as f:
                    content: str = base64.b64encode(f.read())
                    ret_img = FileBox.from_base64(name=os.path.basename(ret_path), base64=content)
                    await msg.say(ret_img)
