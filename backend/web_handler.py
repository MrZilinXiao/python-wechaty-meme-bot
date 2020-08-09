import string
import random

from flask import Flask, request, jsonify
from backend.response.dispatcher import RequestDispatcher
import backend.config as config
import os
import base64

meme_bot_front = Flask(__name__)


class WebHandler(object):
    def __init__(self):
        if not os.path.exists(config.history_meme_path):
            os.mkdir(config.history_meme_path)
        self.dispatcher = RequestDispatcher()
        self.history_path = {}

    #     self._walk_history()
    #
    # def _walk_history(self):
    #     for filename in os.listdir(config.history_meme_path):
    #         self.history_path[filename] = True

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
                return jsonify({'status': 401, 'reason': 'image not supported yet!'})
            # decode b64data and save in history path
            data_buf = base64.b64decode(img_b64data)
            save_path = self._check_filename(img_name)
            with open(save_path, 'wb') as f:
                f.write(data_buf)


        except KeyError as ke:
            print(ke)
