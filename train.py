# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score
import wandb

#解决wandb创建./config权限问题
os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"

train_path = "./data/divorce/train_dataset.json"
valid_path = "./data/divorce/test_dataset.json"
labels_path = "./data/divorce/labels.json"
save_model_path = "./model/multi_label_cls.pth"
labels = load_json(labels_path)
class_num = len(labels)
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 128
max_len = 128
hidden_size = 768
epochs = 10

wandb.init(project="multi_label_calssification", config = {"lr":lr,"batch_size":batch_size,"max_len":max_len,"epochs":epochs}, save_code=True)


train_dataset = MultiClsDataSet(train_path,labels_path=labels_path, max_len=max_len)
valid_dataset = MultiClsDataSet(valid_path,labels_path=labels_path, max_len=max_len)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_true_tensor = y_true_tensor.cpu().numpy()
    return accuracy_score(y_true_tensor, y_pred_tensor)


def train_process():
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    valid_best_acc = 0.
    for epoch in range(1, epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                acc_score = get_acc_score(labels, logits)
                print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))
                wandb.log({"step":i,"step_loss":loss.item(),"step_acc":acc_score})

        # 验证集合
        valid_loss, valid_acc = valid_process(model, valid_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss:{}".format(epoch, valid_acc, valid_loss))
        wandb.log({"epoch":epoch,"valid_loss":valid_loss,"valid_acc":valid_acc})
        if valid_acc > valid_best_acc:
            valid_best_acc = valid_acc
            torch.save(model.state_dict(), save_model_path)

    # 测试
    #test_acc = test(save_model_path, test_path)
    #print("Test acc: {}".format(test_acc))
    wandb.finish()


def valid_process(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test_process(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path,valid_path, max_len=max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return acc_score


if __name__ == '__main__':
    train_process()
