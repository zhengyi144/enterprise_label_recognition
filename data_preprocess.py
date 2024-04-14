# -*- coding: utf-8 -*-

"""
数据预处理
"""
import os
import json
from sklearn.model_selection import train_test_split


def load_json(data_path):
    """
    一次性读取json
    """
    with open(data_path, encoding="utf-8") as f:
        return json.loads(f.read())


def dump_json(project, out_path):
    """
    一次性写入json
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False)
    
def read_txt(data_path):
    items=[]
    file=open(data_path, encoding="utf-8")
    for row in file.readlines():
        items.append(row.strip("\n"))
    return items


def divorce_preprocess(data_path,out_path,train_test_ratio=0.2,filter_none_label=True):
    """
    divorce数据预处理:
    划分训练、测试样本数据
    """
    #提取所有样本数据
    sample_data=[]
    labels=[]
    items=read_txt(data_path)
    for item in items:
        item=eval(item)
        #print(item)
        for obj in item:
            #print(obj)
            if filter_none_label:
                if len(obj["labels"])>0:
                    labels.extend(obj["labels"])
                    sample_data.append({"labels":obj["labels"],"text":obj["sentence"]})
            else:
                sample_data.append(obj)
    #划分训练、测试集合
    #random_state:设置随机种子，保证每次运行生成相同的随机数
    #test_size:将数据分割成训练集的比例
    train_set, test_set = train_test_split(sample_data, test_size=train_test_ratio, random_state=42)

    #写训练集及测试
    with open(os.path.join(out_path,"train_dataset.json"), "w", encoding="utf-8") as f:
        for train_data in train_set:
            f.write(json.dumps(train_data, ensure_ascii=False) + "\n")
    with open(os.path.join(out_path,"valid_dataset.json"), "w", encoding="utf-8") as f:
        for test_data in test_set:
            f.write(json.dumps(test_data, ensure_ascii=False) + "\n")

    #写入标签
    labels = list(set(labels))
    labels_idx = {label: idx for idx, label in enumerate(labels)}
    dump_json(labels_idx, os.path.join(out_path,"labels.json"))


def estimate_text_max_length(train_data_path, max_len_ratio=0.9):
    """
    :param train_data_path:
    :param labels_idx_path:
    :param max_len_ratio:
    :return:
    """
    text_length = []
    with open(train_data_path, encoding="utf-8") as f:
        for data in f:
            data = json.loads(data)
            text_length.append(len(data["text"]))
    text_length.sort()

    print("当设置max_len={}时，可覆盖{}的文本".format(text_length[int(len(text_length)*max_len_ratio)], max_len_ratio))


if __name__ == '__main__':
    #divorce_preprocess("./data/divorce/data_small_selected.txt","./data/divorce/")
    estimate_text_max_length("./data/divorce/train_dataset.json", max_len_ratio=0.9)
    
