# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        self.bert = BertModel.from_pretrained("./pretrain_model/bert_base_chinese/")

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls = self.drop(outputs[1])
        out = F.sigmoid(self.fc(cls))
        return out

