import os
from tkinter import N
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from models.fudmend_get_domain import MultiDomainFENDModel
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

path1 = Path('/home/yanzhou/MDFEND_all_user/data')
path2 = Path('/home/yanzhou/MDFEND_all_user/state_dict/encode')
path3 = Path('/home/yanzhou/NewModel/ProcessData')
split = [[], [], []]  # 划分训练集、测试集， 11826
# split_list = pd.read_csv('/home/yanzhou/BIC-main/datasets/Twibot-20/split.csv')  # （229580，2）
# label = pd.read_csv('/home/yanzhou/BIC-main/datasets/Twibot-20/label.csv')  # （11826，2）用户ID，标签
split_list = pd.read_csv('/home/yanzhou/Data/Twibot-20/split.csv')  # （229580，2）
label = pd.read_csv('/home/yanzhou/MDFEND_all_user/data/label.csv')  # （11826，2）用户ID，标签
domain_path = Path('/home/yanzhou/MDFEND_all_user_graph')
torch.manual_seed(2233)
# 获取训练集、测试集、验证集用户id
users_index_to_uid = list(label['id'])  # 11826
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}  # 字典 11826
for id in split_list[split_list['split'] == 'train']['id']:  #
    split[0].append(uid_to_users_index[id])  # 用户id转换为index
for id in split_list[split_list['split'] == 'val']['id']:
    split[1].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'test']['id']:
    split[2].append(uid_to_users_index[id])
BATCH_SIZE = 64


def record_highest(num_added, in_list):
    for i in range(len(in_list)):
        if num_added > in_list[i]:
            if i == 0:
                in_list[i] == num_added
            else:
                in_list[i - 1] = in_list[i]
                in_list[i] = num_added


def eval(all_confusion):
    acc = (all_confusion[0][0] + all_confusion[1][1]) / np.sum(all_confusion)
    precision = all_confusion[1][1] / (all_confusion[1][1] + all_confusion[0][1])
    recall = all_confusion[1][1] / (all_confusion[1][1] + all_confusion[1][0])
    f1 = (2 * precision * recall) / (precision + recall)
    return acc, f1, precision, recall


def my_collate(batch):  # len(batch[0]) : 8
    # text （batch数据） 推文+描述    # 能得到 text[index], user_neighbor_index[index],
    text = torch.stack([item[0] for item in batch])  # torch.Size([64, 201, 768])  batch: <class 'list'>
    # user_label：
    # category = torch.stack([item[1] for item in batch]).type(torch.LongTensor)
    domain = torch.stack([item[1] for item in batch])
    user_neighbor_index = [item[2] for item in batch]  # <class 'list'>, 中心节点的邻居节点
    user_label = torch.stack([item[3] for item in batch]).type(torch.LongTensor)  # 64
    num_feature = batch[0][4]  # torch.Size([229580, 4])
    cat_feature = batch[0][5]  # torch.Size([229580, 2])
    edge_index = batch[0][6]  # torch.Size([2, 455958])
    return [text, domain, user_neighbor_index, user_label, num_feature, cat_feature, edge_index]


class InterDataset(Dataset):
    def __init__(self, name='train'):
        super(InterDataset, self).__init__()
        self.text = torch.load(path1 / 'text.pt')  # 11826 torch.Size([11826, 201, 768])
        self.category = torch.load(path1 / 'category.pt')
        self.user_neighbor_index = np.load(path1 / 'user_neighbor_index_1.npy',
                                           allow_pickle=True).tolist()  # 11826, 与每个用户相关联的邻居节点
        self.domain = torch.load(domain_path / 'tweets_domain_tensor.pt')
        self.user_label = torch.load(path1 / 'node_label.pt')  # torch.Size([11826])
        self.num_feature = torch.load(path1 / 'num1.pt')  # 229580 torch.Size([229580, 4])
        self.cat_feature = torch.load(path1 / 'cat1.pt')  # 229580 torch.Size([229580, 2])
        self.edge_index = torch.load(path1 / 'edge_index_1.pt')  # check if tensor, torch.Size([2, 455958])

        if name == 'train':
            self.text = self.text[split[0]]  # torch.Size([8278, 201, 768])
            self.category = self.category[split[0]]
            self.user_neighbor_index = [self.user_neighbor_index[split[0][i]] for i in range(len(split[0]))]  # 8278
            self.user_label = self.user_label[split[0]]  # torch.Size([8278])
            self.domain = self.domain[split[0]]
            self.length = len(self.user_label)  # 8278
        if name == 'val':
            self.text = self.text[split[1]]  # torch.Size([2365, 201, 768])
            self.category = self.category[split[1]]
            self.user_neighbor_index = [self.user_neighbor_index[split[1][i]] for i in range(len(split[1]))]  # 2365
            self.user_label = self.user_label[split[1]]  # torch.Size([2365])
            self.domain = self.domain[split[1]]
            self.length = len(self.user_label)  # 2365
        if name == 'test':
            self.text = self.text[split[2]]
            self.category = self.category[split[2]]
            self.user_neighbor_index = [self.user_neighbor_index[split[2][i]] for i in range(len(split[2]))]
            self.user_label = self.user_label[split[2]]
            self.domain = self.domain[split[2]]
            self.length = len(self.user_label)

    def __len__(self):
        return self.length  # 8278, 2365, 1183

    def __getitem__(self, index):
        return self.text[index], self.domain[index], self.user_neighbor_index[index], \
               self.user_label[index], self.num_feature, self.cat_feature, self.edge_index


state_dict_path = '/home/yanzhou/MDFEND_all_user_graph/state_dict/encode/state_dict_model.tar'
best_model = torch.load(state_dict_path)
print(best_model.get('f1'))
print(best_model.get('acc'))
model_param = best_model.get('model_state_dict')
dropout = 0.5
model = MultiDomainFENDModel(emb_dim=768, mlp_dims=[384], dropout=dropout)  # InteractModel
model.load_state_dict(model_param)
train_dataset = InterDataset('train')
val_dataset = InterDataset('val')
test_dataset = InterDataset('test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
device = 'cuda:0'
model = model.cuda()
model.eval()
domain_list = None
with torch.no_grad():
    for batch in train_loader:
        text = batch[0].to(device)
        domain = batch[1].to(device)
        user_neighbor_index = batch[2]
        user_label = batch[3].numpy()
        num_feature = batch[4].to(device)
        cat_feature = batch[5].to(device)
        edge_index = batch[6].to(device)
        pred, domain = model(text, domain, user_neighbor_index, num_feature, cat_feature, edge_index)
        # print(domain.shape)
        domain = domain.cpu().numpy()
        if domain_list is None:
            domain_list = domain
        else:
            domain_list = np.vstack((domain_list, domain))
    for batch in val_loader:
        text = batch[0].to(device)
        domain = batch[1].to(device)
        user_neighbor_index = batch[2]
        user_label = batch[3].numpy()
        num_feature = batch[4].to(device)
        cat_feature = batch[5].to(device)
        edge_index = batch[6].to(device)
        pred, domain = model(text, domain, user_neighbor_index, num_feature, cat_feature, edge_index)
        # print(domain.shape)
        domain = domain.cpu().numpy()
        domain_list = np.vstack((domain_list, domain))
    for batch in test_loader:
        text = batch[0].to(device)
        domain = batch[1].to(device)
        user_neighbor_index = batch[2]
        user_label = batch[3].numpy()
        num_feature = batch[4].to(device)
        cat_feature = batch[5].to(device)
        edge_index = batch[6].to(device)
        pred, domain = model(text, domain, user_neighbor_index, num_feature, cat_feature, edge_index)
        # print(domain.shape)
        domain = domain.cpu().numpy()
        domain_list = np.vstack((domain_list, domain))
np.save("gate_value.npy", domain_list)
