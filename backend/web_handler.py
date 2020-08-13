import string
import random
from abc import ABC

from flask import Flask, request, jsonify, views
from backend.response.dispatcher import RequestDispatcher
import backend.config as config
import os
import base64
from PIL import Image
from orm import History
import hashlib


class WebHandler:
    dispatcher = None
    history_path = {}
    exception_dir = {
        400: 'request method not correct!',
        401: 'image not supported yet!',
        402: 'backend meme image missing!'
    }

    def __init__(self):
        if not os.path.exists(config.history_meme_path):
            os.mkdir(config.history_meme_path)
        if self.dispatcher is None:
            self.dispatcher = RequestDispatcher()
        # self.history_path = {}

    @staticmethod
    def exception(code: int) -> dict:
        return {'status': code, 'reason': WebHandler.exception_dir[code]}

    @staticmethod
    def _generate_random_str(length=16):
        str_list = [random.choice(string.digits + string.ascii_letters) for i in range(length)]
        random_str = ''.join(str_list)
        return random_str

    def check_filename(self, filename: str) -> str:
        intended_name = os.path.join(config.history_meme_path, filename)
        if not os.path.exists(intended_name):
            return intended_name
        else:
            # loop until find a suitable random name
            return self.check_filename(WebHandler._generate_random_str() +
                                       '.' + filename.split(os.path.sep)[-1])


web_handler = WebHandler()


class WebView(views.View, ABC):
    methods = ['GET', 'POST']

    def __init__(self):
        super(WebView, self).__init__()

    def dispatch_request(self):
        if request.method != 'POST':
            return jsonify(web_handler.exception(400))
        try:
            img_name, img_b64data = request.form.get('img_name'), request.form.get('data')  # type: str, str
            if not img_name.endswith(config.allow_img_extensions):
                return jsonify(web_handler.exception(401))
            # decode b64data and save in history path
            data_buf = base64.b64decode(img_b64data)
            save_path = web_handler.check_filename(img_name)
            with open(save_path, 'wb') as f:
                f.write(data_buf)
            ret_meme, log_list = web_handler.dispatcher.receive_handler(save_path)  # type: str, list
            md5_str = hashlib.md5(open(ret_meme, 'rb').read()).hexdigest()

            History.create(receive_img_path=save_path, response_log='\n'.join(log_list),
                           response_img_path=ret_meme)
            return jsonify({
                'img_name': '/' + '/'.join(ret_meme.split('/')[-2:]),  # for frontend to perform static reference
                # suffix of URL depends on frontend setting, making it possible to get CDN support
                'md5': md5_str,
                'log': '\n'.join(log_list)
            })

        except KeyError as ke:  # raise KeyError when meeting abnormal request
            print(ke)


webview = WebView()

meme_bot_front = Flask(__name__, static_url_path=config.static_url_path,
                       static_folder=config.static_folder)
meme_bot_front.add_url_rule(config.backend_post_url, view_func=webview.as_view('upload'))

if __name__ == '__main__':
    meme_bot_front.run(host='0.0.0.0', debug=True)
