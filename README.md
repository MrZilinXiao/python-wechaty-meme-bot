# Wechaty-Meme-Bot [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

![meme-bot-logo](/image/logo.png)

[![Wechaty in Python](https://img.shields.io/badge/Wechaty-Python-blue)](https://github.com/wechaty/python-wechaty)
[![Powered by Wechaty](https://img.shields.io/badge/Powered%20By-Wechaty-brightgreen.svg)](https://github.com/Wechaty/wechaty)
## Preface
**This project is supported by [Wechaty Community](https://github.com/wechaty) and [Institute of Software of Chinese Academy of Sciences](https://isrc.iscas.ac.cn/summer2020/).**
[![Wechaty](/image/wechaty-logo.svg)](https://github.com/wechaty/wechaty)
[![summerofcode](/image/summer2020.svg)](https://isrc.iscas.ac.cn/summer2020/)

PowerPoint Demonstration: [https://www.bilibili.com/video/BV1kZ4y1M7F6/](https://www.bilibili.com/video/BV1kZ4y1M7F6/)

Demo Live Video on bilibili: [https://www.bilibili.com/video/BV17f4y197ut/](https://www.bilibili.com/video/BV17f4y197ut/)

My community mentor is [Huang](https://github.com/huangaszaq), contributor of python-wechaty. I won't make such progress without his support.

## Introduction
I was required to build a meme bot based on [python-wechaty](https://github.com/wechaty/python-wechaty), which should possess at least following functions:
- receive & save meme image from specific contact
- analyse received meme image 
- response meme image accordingly based on analysis given above

To achieve such requirements, I came out with a cross-functional diagram below to assist my development(written in Chinese):
![](https://upyun.mrxiao.net/img/Drawing4.svg)

## Some Concepts
1. Frontend: Run on user end, be in charge of communicating with python-wechaty-puppet and backend, acting as a middleware.
2. Backend: Run on server end equipping a NVIDIA GPU, be in charge of analyzing meme image and choose response meme based on certain strategy.

## Directory Layout

```shell script
$ tree -L 3 -I '__pycache__'
.
├── LICENSE
├── Makefile
├── README.md
├── backend  # backend files
│   ├── chineseocr_lite  # modified OCR module
│   │   ├── Dockerfile
│   │   ├── LICENSE
│   │   ├── __init__.py
│   │   ├── angle_class
│   │   ├── config.py
│   │   ├── crnn
│   │   ├── model.py
│   │   ├── models
│   │   ├── psenet
│   │   └── utils.py
│   ├── config.yaml   # config file in yaml format
│   ├── conversation  # conversation GPT2 model path (~600MB), download from GDrive mentioned before
│   ├── cosine_metric_net.py  # definition of CosineMetricNet
│   ├── cosine_train  # CosineMetricNet Training scripts
│   │   ├── dataset.py
│   │   ├── metric.py
│   │   └── train_and_eval.py
│   ├── dataset.py  # Common training dataset module
│   ├── feature_extract.py  # feature extract module
│   ├── hanlp_wrapper.py  # NLP wrapper
│   ├── logs  # log dir
│   ├── meme  # default dir for meme import
│   │   ├── classified
│   │   ├── others
│   │   └── unclassified
│   ├── meme_importer.py
│   ├── ocr_wrapper.py
│   ├── requirements.txt
│   ├── response
│   │   ├── __init__.py
│   │   ├── conversation.py
│   │   ├── dispatcher.py
│   │   └── feature.py
│   ├── spider  # custom spider dir, any spiders should derive from BaseSpider
│   │   ├── BaseSpider.py
│   │   └── FaBiaoQingSpider.py  # example spider to crawl FaBiaoQing
│   ├── stopwords.txt  # stop words list for NLP tokenizer
│   ├── utils.py  # backend public utils
│   └── web_handler.py  # backend Flask module
├── frontend
│   ├── config.py  # frontend configuration
│   ├── image  # image cache dir
│   ├── logs  # log dir
│   ├── main.py
│   └── meme_bot.py
├── gdrive.sh   # bash to download from GDrive
├── image  # static image files
│   ├── logo.png
│   ├── summer2020.svg
│   └── wechaty-logo.svg
├── orm.py  # orm module
├── test.db   # SQLite database
└── tests  # unittests using pytest
    ├── conftest.py
    ├── test_conversation.py
    ├── test_dataset.py
    └── test_orm.py
```

## Deploy Tutorial
```
git clone https://github.com/MrZilinXiao/python-wechaty-meme-bot.git
```
### Frontend
#### Via PyPi (Pending...)
```shell script
pip3 install wechaty-meme-bot
```
 
#### Manually

1.Correctly configure backend settings in `frontend/config.yaml`
```yaml
general:
  image_temp_dir: './image'
  allow_img_extensions: ('.jpg', '.png', '.jpeg', '.gif')

backend:  # change to your backend server
  backend_upload_url: 'http://192.168.10.102:5000/meme/upload'
  backend_static_url: 'http://192.168.10.102:5000/static'
```
2.Run lines below in your shell:
```shell script
export WECHATY_PUPPET=wechaty-puppet-hostie
export WECHATY_PUPPET_HOSTIE_TOKEN=your-donut-token   # replace `your-donut-token` with your wechaty donut token
make run-frontend
# if no `make` in your system, try run `pip3 install -r frontend/requirements.txt`, `python3 frontend/main.py`
```

### Backend
*Currently we only get backend tested on Ubuntu, while frontend possesses cross-platform feature.*

You may refer to [Github Action Configuration](https://github.com/MrZilinXiao/python-wechaty-meme-bot/blob/master/.github/workflows/test.yml) to learn how we deploy backend when you encounter issues.
#### Nvidia-docker
Pending...

#### Manually
```shell script
pip3 install -r backend/requirements.txt
python backend/web_handler.py  # this will trigger chineseocrlite compiling process
```

## Open-Source Reference
- [chineseocr_lite](https://github.com/ouyanghuiyu/chineseocr_lite/tree/master): Powerful Chinese OCR module with accurate results and fast inference.
- [HaNLP](https://github.com/hankcs/HanLP): Multilingual NLP library for researchers and companies, built on TensorFlow 2.0.
- [Transformers](https://github.com/huggingface/transformers): State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese): Chinese version of GPT2 training code, using BERT tokenizer.

## Academic Citation
```
# in backend/cosine_metric_net.py
[1]N. Wojke and A. Bewley, “Deep Cosine Metric Learning for Person Re-identification,” in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, Mar. 2018, pp. 748–756, doi: 10.1109/WACV.2018.00087.
# GPT2 Original Paper
[2]Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019): 9.
```
