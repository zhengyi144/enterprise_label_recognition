# -*- coding: utf-8 -*-

import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer

hidden_size = 768
label2idx_path = "./data/enterprise_tags/labels.json"
save_model_path = "./model/multi_label_cls_v2.pth"
label2idx = load_json(label2idx_path)
class_num=len(label2idx)
idx2label = {idx: label for label, idx in label2idx.items()}
print(idx2label,class_num)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("./pretrain_model/bert_base_chinese/")
max_len = 256

model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
model.load_state_dict(torch.load(save_model_path,map_location=device))
model.to(device)
model.eval()


def text_batched(text_list):
    batchs=[]
    batch=''
    for item in text_list:
        if item =="":
            continue
        if len(batch)<max_len:
            batch+=item+";"
        else:
            batchs.append(batch)
            batch=item+";"
    batchs.append(batch)
    return batchs

def predict(texts):
    outputs = tokenizer(texts, return_tensors="pt", max_length=max_len,
                        padding=True, truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device),
                   outputs["token_type_ids"].to(device))
    logits = logits.cpu().tolist()
    # print(logits)
    result = []
    for sample in logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                pred_label.append(idx2label[idx])
        result.append(pred_label)
    return result


if __name__ == '__main__':
    texts = ["一种智能门禁控制系统;一种基于深度学习的显示面板外观检测方法;", "今日沪深两市指数整体呈现震荡调整格局"]
    result = predict(texts)
    print(result)


