# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_preprocess import load_json


class MultiClsDataSet(Dataset):
    def __init__(self, data_path, labels_path, max_len=510):
        self.labels = load_json(labels_path)
        print(self.labels)
        self.class_num = len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained("./pretrain_model/bert_base_chinese/")
        self.max_len = max_len
        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.encoder(data_path)

    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                texts.append(line["text"])
                tmp_label = [0] * self.class_num
                for label in line["labels"]:
                    tmp_label[self.labels[label]] = 1
                labels.append(tmp_label)

        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, token_type_ids, attention_mask, \
               torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input_ids[item],  self.attention_mask[item], \
               self.token_type_ids[item], self.labels[item]


if __name__ == '__main__':
    dataset = MultiClsDataSet(data_path="./data/enterprise_tags/train_dataset.json",labels_path="./data/enterprise_tags/labels.json")
    print(dataset.input_ids)
    print(dataset.token_type_ids)
    print(dataset.attention_mask)
    print(dataset.labels)