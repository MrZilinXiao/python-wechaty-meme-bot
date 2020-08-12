import string
import random

from flask import Flask, request, jsonify
from backend.response.dispatcher import RequestDispatcher
import backend.config as config
import os
import base64
from PIL import Image
from orm import History
import hashlib

meme_bot_front = Flask(__name__, static_url_path=config.static_url_path,
                       static_folder=config.static_folder)


class WebHandler(object):
    exception_dir = {
        401: 'image not supported yet!'
    }

    def __init__(self):
        if not os.path.exists(config.history_meme_path):
            os.mkdir(config.history_meme_path)
        self.dispatcher = RequestDispatcher()
        self.history_path = {}

    @staticmethod
    def _exception(code: int) -> dict:
        return {'status': code, 'reason': WebHandler.exception_dir[code]}

    @staticmethod
    def _generate_random_str(length=16):
        str_list = [random.choice(string.digits + string.ascii_letters) for i in range(length)]
        random_str = ''.join(str_list)
        return random_str

    def _check_filename(self, filename: str) -> str:
        intended_name = os.path.join(config.history_meme_path, filename)
        if not os.path.exists(intended_name):
            return intended_name
        else:
            # loop until find a suitable random name
            return self._check_filename(WebHandler._generate_random_str() +
                                        '.' + filename.split(os.path.sep)[-1])

    @meme_bot_front.route(config.backend_post_url, methods=['POST'])
    def upload(self):
        try:
            img_name, img_b64data = request.form.get('img_name'), request.form.get('data')  # type: str, str
            if not img_name.endswith(config.allow_img_extensions):
                return jsonify(WebHandler._exception(401))
            # decode b64data and save in history path
            data_buf = base64.b64decode(img_b64data)
            save_path = self._check_filename(img_name)
            with open(save_path, 'wb') as f:
                f.write(data_buf)
            ret_meme, log_list = self.dispatcher.receive_handler(save_path)  # type: str, list
            md5_str = hashlib.md5(open(ret_meme, 'rb').read()).hexdigest()

            History.create(receive_img_path=save_path, response_log='\n'.join(log_list),
                           response_img_path=ret_meme)
            return jsonify({
                'img_name': '/' + '/'.join(ret_meme.split('/')[-2:]),  # for frontend to perform static reference
                # suffix of URL depends on frontend setting, making it possible to get CDN support
                'md5': md5_str
            })

        except KeyError as ke:  # raise KeyError when meeting abnormal request
            print(ke)
