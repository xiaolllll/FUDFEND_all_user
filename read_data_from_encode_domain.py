import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from models.encode_fudmend import MultiDomainFENDModel
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

path1 = Path('/home/yanzhou/MDFEND_all_user/data')
path2 = Path('/home/yanzhou/MDFEND_all_user/state_dict')
path3 = Path('/home/yanzhou/NewModel/ProcessData')
split = [[], [], []]  # 划分训练集、测试集， 11826
split_list = pd.read_csv('/home/yanzhou/Data/Twibot-20/split.csv')  # （229580，2）
label = pd.read_csv('/home/yanzhou/MDFEND_all_user/data/label.csv')  # （11826，2）用户ID，标签
domain_path = Path('/home/yanzhou/MDFEND_all_user_graph')

# 获取训练集、测试集、验证集用户id
users_index_to_uid = list(label['id'])  # 11826
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}  # 字典 11826
for id in split_list[split_list['split'] == 'train']['id']:  #
    split[0].append(uid_to_users_index[id])  # 用户id转换为index
for id in split_list[split_list['split'] == 'val']['id']:
    split[1].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'test']['id']:
    split[2].append(uid_to_users_index[id])


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


class InterDataset(Dataset):
    def __init__(self, name='train'):
        super(InterDataset, self).__init__()
        self.text = torch.load(path1 / 'text.pt')  # 11826 torch.Size([11826, 201, 768])
        self.category = torch.load(path1 / 'category.pt')
        self.user_neighbor_index = np.load(path1 / 'user_neighbor_index_1.npy',
                                           allow_pickle=True).tolist()  # 11826, 与每个用户相关联的邻居节点
        tmp = np.load('domain_encode_after_norm.npy')
        self.domain = torch.Tensor(tmp)
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


# 合并一个样本列表，形成一个张量(s)的小批。当从映射风格的数据集中使用批处理加载时使用,,主要是用它合成，因为data类型较多
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


def get_metric(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    f_score = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return auc, f_score, recall, precision, acc


BATCH_SIZE = 64


class InterTrainer:
    def __init__(self, train_loader, val_loader, test_loader, model=MultiDomainFENDModel, optimizer=torch.optim.Adam,
                 lr=1e-4, weight_decay=1e-5, scheduler=ReduceLROnPlateau, dropout=0.5, num_epochs=100,
                 early_stopping_patience=20,
                 state_dict_path='state_dict_only_graph.tar', device='cuda:0'):
        self.device = device
        self.state_dict_path = state_dict_path  # 'state_dict.tar'
        self.num_epochs = num_epochs  # 30
        self.model = model(emb_dim=768, mlp_dims=[384], dropout=dropout)  # InteractModel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = scheduler(self.optimizer, mode='max', verbose=True, factor=0.1, patience=5, eps=1e-8)  # 优化器
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)  # 早停
        self.loss_func = nn.BCELoss()

        self.highest_5_acc = [0, 0, 0, 0, 0]
        self.highest_5_f1 = [0, 0, 0, 0, 0]

    def train(self):
        max_acc, corres_f1, model_state_dict = 0, 0, None
        epoch_loss = 0
        self.model = self.model.cuda()
        for epoch in range(self.num_epochs):
            self.model.train()
            batch_quantity = 0
            with tqdm(self.train_loader) as progress_bar:
                for batch in progress_bar:
                    text = batch[0].to(self.device)  # torch.Size([64, 201, 768])
                    domain = batch[1].to(self.device)
                    user_neighbor_index = batch[2]
                    user_label = batch[3].to(self.device)
                    num_feature = batch[4].to(self.device)
                    cat_feature = batch[5].to(self.device)
                    edge_index = batch[6].to(self.device)
                    pred = self.model(text, domain, user_neighbor_index, num_feature, cat_feature,
                                      edge_index)  # torch.Size([64, 2])
                    # print(pred)
                    loss = self.loss_func(pred, user_label.float())
                    epoch_loss += loss.item()
                    batch_quantity += 1

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())

            epoch_loss /= batch_quantity
            print(f'epoch_loss={epoch_loss}')

            val_loss, val_acc = self.val()
            self.scheduler.step(val_acc)
            test_acc, corres_f1, recall, precision, model_state_dict = self.test()

            print(f'Test_Accuracy: {test_acc}', end=' ')
            print(f'Precision:{precision}', end=' ')
            print(f'Recall:{recall}', end=' ')
            print(f'F1:{corres_f1}')

            if max_acc < val_acc:
                a = open('/home/yanzhou/MDFEND_all_user/data/result_model.txt', 'a')
                a.write('test_acc：' + str(test_acc) + '\n')
                a.write('recall：' + str(recall) + '\n')
                a.write('Precision：' + str(precision) + '\n')
                a.write('corres_f1：' + str(corres_f1) + '\n\n')
                a.close()

                max_acc = val_acc
                if not os.path.exists(path2 / self.state_dict_path):
                    torch.save({'acc': test_acc, 'f1': corres_f1, 'recall': recall, 'Precision': precision,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'model_state_dict': model_state_dict}, path2 / self.state_dict_path)
                else:
                    best_model = torch.load(path2 / self.state_dict_path)
                    if best_model['acc'] < max_acc:
                        torch.save({'acc': test_acc, 'f1': corres_f1, 'recall': recall, 'Precision': precision,
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'model_state_dict': model_state_dict}, path2 / self.state_dict_path)

    @torch.no_grad()
    def val(self):
        self.model.eval()
        batch_quantity, epoch_loss = 0, 0
        all_confusion = np.zeros([2, 2])
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch in self.val_loader:
                text = batch[0].to(self.device)  # torch.Size([64, 201, 768])
                domain = batch[1].to(self.device)
                user_neighbor_index = batch[2]
                user_label = batch[3].to(self.device)
                num_feature = batch[4].to(self.device)
                cat_feature = batch[5].to(self.device)
                edge_index = batch[6].to(self.device)
                pred = self.model(text, domain, user_neighbor_index, num_feature, cat_feature, edge_index)
                loss = self.loss_func(pred, user_label.float())
                pred_list.extend(pred.cpu().numpy().tolist())
                label_list.extend(user_label.cpu().numpy().tolist())
                # pred, user_label = pred.argmax(dim=-1).cpu().numpy(), user_label.cpu().numpy()
                epoch_loss += loss
                batch_quantity += 1
                # all_confusion += confusion_matrix(user_label.float(), pred)

            epoch_loss /= batch_quantity
            auc, f1, recall, precision, acc = get_metric(label_list, pred_list)
            # acc, f1, precision, recall = eval(all_confusion)
            print(f'Val_Accuracy: {acc}', end=' ')
            print(f'Precision:{precision}', end=' ')
            print(f'Recall:{recall}', end=' ')
            print(f'F1:{f1}')
            print(f'val_loss:{epoch_loss}')

        a = open('/home/yanzhou/MDFEND_all_user/data/result_model.txt', 'a')
        a.write('epoch_loss：' + str(epoch_loss) + '\n')
        a.write('val acc：' + str(acc) + '\n')
        a.write('val Recall：' + str(recall) + '\n')
        a.write('val F1：' + str(f1) + '\n')
        a.write('Val Precision：' + str(precision) + '\n')
        a.close()

        return epoch_loss, acc

    @torch.no_grad()
    def test(self):
        self.model.eval()
        all_confusion = np.zeros([2, 2])
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch in self.test_loader:
                text = batch[0].to(self.device)
                domain = batch[1].to(self.device)
                user_neighbor_index = batch[2]
                user_label = batch[3].numpy()
                num_feature = batch[4].to(self.device)
                cat_feature = batch[5].to(self.device)
                edge_index = batch[6].to(self.device)
                pred = self.model(text, domain, user_neighbor_index, num_feature, cat_feature, edge_index).cpu().numpy()
                pred_list.extend(pred.tolist())
                label_list.extend(user_label.tolist())
                # all_confusion += confusion_matrix(user_label.float(), pred)

            # acc, f1, precision, recall = eval(all_confusion)
            auc, f1, recall, precision, acc = get_metric(label_list, pred_list)
            record_highest(acc, self.highest_5_acc)
            record_highest(f1, self.highest_5_f1)

        model_state_dict = self.model.state_dict()

        return acc, f1, recall, precision, model_state_dict


if __name__ == '__main__':
    train_dataset = InterDataset('train')
    val_dataset = InterDataset('val')
    test_dataset = InterDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)

    trainer = InterTrainer(train_loader, val_loader, test_loader, model=MultiDomainFENDModel,
                           optimizer=torch.optim.RAdam,
                           lr=1e-4, weight_decay=1e-5, scheduler=ReduceLROnPlateau, dropout=0.5, num_epochs=100,
                           early_stopping_patience=10,
                           state_dict_path='/home/yanzhou/MDFEND_all_user/state_dict/state_dict_model.tar',
                           device='cuda:0')
    trainer.train()
