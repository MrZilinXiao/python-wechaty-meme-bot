# 基于python-wechaty建立一个斗图机器人 python-wechaty-meme-bot
## Preface
**This project is supported by [Wechaty Community](https://github.com/wechaty) and [Institute of Software of Chinese Academy of Sciences](https://isrc.iscas.ac.cn/summer2020/).**
[![Wechaty](https://wechaty.js.org/img/wechaty-logo.svg)](https://github.com/wechaty/wechaty)
[![summerofcode](https://isrc.iscas.ac.cn/summer2020/help/assets/summer2020.svg)](https://isrc.iscas.ac.cn/summer2020/)

PowerPoint Demostration: [https://www.bilibili.com/video/BV1kZ4y1M7F6/](https://www.bilibili.com/video/BV1kZ4y1M7F6/)

Demo Live Video on bilibili: [https://www.bilibili.com/video/BV17f4y197ut/](https://www.bilibili.com/video/BV17f4y197ut/)

My community mentor is [Huang](https://github.com/huangaszaq), contributor of python-wechaty. I won't make such progress without his support.

## Introduction
I was required to build a meme bot based on [python-wechaty](https://github.com/wechaty/python-wechaty), which should possess following functions:
- receive&save meme image from specific contact
- analyse received meme image 
- response meme image accordingly based on analysis given above

To achieve such requirements, I came out with a cross-functional diagram below to assist my development(written in Chinese):
![](https://upyun.mrxiao.net/img/Drawing4.svg)

## Some Concepts
1. Frontend: Run on user end, be in charge of communicating with python-wechaty-puppet and backend, act as a middleware.
2. Backend: Run on server end equipping a NVIDIA GPU, be in charge of analyzing meme image and choose response meme based on certain strategy.

## Directory Layout

```shell script
$ tree -L 3 -I '__pycache__'
.
├── LICENSE
├── README.md
├── SCHEDULE.md
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
│   ├── config.py  # backend configuration
│   ├── cosine_metric_net.py  # definition of CosineMetricNet
│   ├── cosine_train  # CosineMetricNet Training scripts
│   │   ├── dataset.py
│   │   ├── metric.py
│   │   └── train_and_eval.py
│   ├── dataset.py  # Common training dataset module
│   ├── feature_extract.py  # feature extract module
│   ├── hanlp_wrapper.py  # NLP wrapper
│   ├── logs  # log dir
│   │   └── __init__.py
│   ├── meme   # meme image dir
│   │   ├── classified
│   │   ├── others
│   │   └── unclassified
│   ├── meme_importer.py
│   ├── ocr_wrapper.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── response
│   │   ├── __init__.py
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
├── orm.py  # orm module
├── test.db   # SQLite database
├── tests
└── unittests.py

```

## Deploy Tutorial
### Docker Deployment
Pending...

### Manual Deployment
```
git clone https://github.com/MrZilinXiao/python-wechaty-meme-bot.git
```
#### Frontend
```shell script
pip install -r ./frontend/requirements.txt
export WECHATY_PUPPET=wechaty-puppet-hostie
export WECHATY_PUPPET_HOSTIE_TOKEN=XXX   # replace with your wechaty donut token
python frontend/main.py
```
#### Backend
Premise: CMake & C++11 compile environment needed.
```shell script
pip install -r ./backend/requirements.txt
python backend/web_handler.py  # this will trigger chineseocrlite compiling process
```

## Open-Source Reference
- [chineseocr_lite](https://github.com/ouyanghuiyu/chineseocr_lite/tree/master): Powerful Chinese OCR module with accurate results and fast inference.
- [HaNLP](https://github.com/hankcs/HanLP): Multilingual NLP library for researchers and companies, built on TensorFlow 2.0.

## Academic Citation
```
# in backend/cosine_metric_net.py
[1]N. Wojke and A. Bewley, “Deep Cosine Metric Learning for Person Re-identification,” in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, Mar. 2018, pp. 748–756, doi: 10.1109/WACV.2018.00087.
```