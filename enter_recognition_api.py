import os
import sys
#print(sys.path)
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#print(sys.path)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
import requests
import json
import random
import logging 
from logging import handlers
from flask import Flask,request
from predict import *
from enter_tag_post import *


app = Flask(__name__)
#初始化日志
format=logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)
time_rotating_file_handler = handlers.TimedRotatingFileHandler(filename='./enter_log/enter_tag_recognise.log',when='D')
time_rotating_file_handler.setLevel(20)
time_rotating_file_handler.setFormatter(format)

logger.addHandler(time_rotating_file_handler)

@app.route("/v1/enter_tag_recognise",methods=["POST"])
def enter_tag_recognise():
    try:
        enterId=request.get_json()["enterId"]
        enterName=request.get_json()["enterName"]
        enterPatents=request.get_json()["enterPatents"]
        enterCopyrights=request.get_json()["enterCopyrights"]
        enterBrief=request.get_json()["enterBrief"]
        
        #剔除重复
        text_set=set()
        for patent in enterPatents:
            if patent!="":
                text_set.add(patent)
        for copyright in enterCopyrights:
            if copyright!="":
                text_set.add(copyright)
        input_texts=list(text_set)
        random.shuffle(input_texts)
        #构建预测text
        batchs=text_batched(input_texts)
        preds=predict(batchs)
        enter_tags=set()
        for pred in preds:
            enter_tags.update(list(pred))
        #后处理
        valid_tags=enter_tag_valid(input_texts,pred_tags=enter_tags)
        logger.critical("enter_tag_recognise enterId:{},enterName:{},input_texts:{},pred_tags:{},valid_tags:{}".format(enterId,enterName,input_texts,preds,valid_tags))

        return {"code":200,"status":"success","message":"企业标签识别成功","data":list(valid_tags)}
    except Exception as e:
        logger.error("enter_tag_recognise error:{}".format(str(e)))
        return {"code":500,"status":"fail","message":"企业标签识别失败"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="enterprise tags recognition Flask API")
    parser.add_argument("--port", default=8090, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port) 