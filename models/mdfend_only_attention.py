from .layers import *
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler
from utils.utils import FixedPooling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 模型交互部分
class InteractModel(nn.Module):
    def __init__(self, num_property_dim=6, cat_property_dim=11, tweet_dim=768,
                 des_dim=768, input_dim=768, hidden_dim=128, output_dim=128,
                 attention_dim=16, graph_num_heads=4, dropout=0.1, device='cuda:0'):
        super(InteractModel, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.attention_dim = attention_dim  # 16
        self.num_linear = nn.Linear(num_property_dim, hidden_dim // 2)  # 4 -- 64
        self.cat_linear = nn.Linear(cat_property_dim, hidden_dim // 2)  # 2--32
        # 图
        self.graph_linear = nn.Linear(hidden_dim, hidden_dim)  # 128 -- 128
        # 文本
        self.text_linear = nn.Linear(input_dim, hidden_dim)  # 768 -- 128

        # 文本传播模块，图传播模块各自部分
        # 输入维度 128， 输出维度 128
        # attention_dim： 16
        # graph_num_heads GCN 作为图神经网络，然后使用多头注意力层利用注意力权重更新用户邻域信息
        self.Model_0 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim,
                                       out_channels=hidden_dim, attention_dim=attention_dim,
                                       graph_num_heads=graph_num_heads, dropout=dropout, device=self.device)
        self.Model_1 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim,
                                       out_channels=hidden_dim, attention_dim=attention_dim,
                                       graph_num_heads=graph_num_heads, dropout=dropout, device=self.device)
        self.Model_2 = RespectiveLayer(in_channels_for_graph=hidden_dim, in_channels_for_text=hidden_dim,
                                       out_channels=output_dim, attention_dim=attention_dim,
                                       graph_num_heads=graph_num_heads, dropout=dropout, device=self.device)
        # 交互表示嵌入
        self.InteractModel_0 = InteractLayer(in_channels=hidden_dim, out_channels=hidden_dim)
        self.InteractModel_1 = InteractLayer(in_channels=hidden_dim, out_channels=hidden_dim)

        #
        self.attention_linear = nn.Linear(attention_dim * attention_dim * 2, output_dim // 3)  # 512 -- 42
        self.user_feature_linear = nn.Linear(output_dim, 132 // 3)  # 128 -- 42
        self.title_linear = nn.Linear(output_dim, output_dim // 3)  # 128 -- 42

        self.final_linear = nn.Linear(42, output_dim)  # 128 -- 128
        self.output = nn.Linear(output_dim, 2)

    def forward(self, text, user_neighbor_index, num_feature, cat_feature, edge_index):
        """
        text: batch_size * 200+1 * 768 torch.Size([64, 201, 768])
        """
        num_feature = self.dropout(self.relu(self.num_linear(num_feature)))
        cat_feature = self.dropout(self.relu(self.cat_linear(cat_feature)))
        all_user_feature = torch.cat((cat_feature, num_feature), dim=-1)
        # all_user_feature 128
        all_user_feature = self.dropout(self.relu(self.graph_linear(all_user_feature)))
        # 全连接，线性层， 768 -- 128
        text = self.text_linear(text)  # torch.Size([64, 201, 128])
        # 文本传播模块，图传播模块
        #  输入维度：
        #  text 128； user_neighbor_index：每个用户节点的邻居节点
        #  all_user_feature： 128
        #  edge_index： torch.Size([2, 455958])# 构成一张图，关注、朋友 # 无向图，源节点、目标节点
        text, all_user_feature, attention_graph_0 = self.Model_0(text, user_neighbor_index,
                                                                 all_user_feature, edge_index)
        # 文本、图 交互模块
        # 输入维度：
        #
        text, all_user_feature = self.InteractModel_0(text, all_user_feature, user_neighbor_index)

        # 文本传播模块，图传播模块
        # 交互表示： text attention_graph_1
        # attention_graph_1 torch.Size([64, 16, 16])
        text, all_user_feature, attention_graph_1 = self.Model_1(text, user_neighbor_index,
                                                                 all_user_feature, edge_index)
        # title 获取第0维全部数据
        title = text[:, 0]
        # 全连接，提取特征
        title = self.dropout(self.relu(self.title_linear(title)))

        # 图交互表示部分
        # user_index = []
        # for neighbor_index in user_neighbor_index:
        #     user_index.append(neighbor_index[0])
        # user_feature = all_user_feature[user_index]

        # user_feature = self.dropout(self.relu(self.user_feature_linear(user_feature)))  # torch.Size([64, 42])
        # 注意力权重 attention_graph_0 torch.Size([64, 16, 16])
        attention_vec_0 = attention_graph_0.view(attention_graph_0.shape[0], self.attention_dim * self.attention_dim)
        attention_vec_1 = attention_graph_1.view(attention_graph_1.shape[0], self.attention_dim * self.attention_dim)
        attention_vec = torch.cat((attention_vec_0, attention_vec_1), dim=-1)  # torch.Size([64, 512])
        # 全连接提取权重特征
        attention_vec = self.dropout(self.relu(self.attention_linear(attention_vec)))  # torch.Size([64, 42])

        # 注意力权重、文本交互、图交互
        final_input = self.final_linear(attention_vec)  # torch.Size([64, 128])

        return final_input


class RespectiveLayer(nn.Module):
    """
    assume LM & GM has same layer
    """

    def __init__(self, in_channels_for_graph=768, in_channels_for_text=768, out_channels=768,
                 attention_dim=6, graph_num_heads=4, text_num_heads=4, dropout=0.5, device='cuda:0'):
        super(RespectiveLayer, self).__init__()
        self.device = device
        # attention_dim 6
        self.attention_dim = attention_dim  # 16

        # 图卷积
        # 图输入维度： 128， 输出维度 128
        self.GCN = GCNConv(in_channels=in_channels_for_graph, out_channels=out_channels)  # 128 -- 128
        # 图多头注意力层，4头
        self.MultiAttn = MultiAttn(embed_dim=out_channels, num_heads=graph_num_heads)  # 128 -- 128
        # 语言模型
        self.LModel = LModel(embed_dim=in_channels_for_text, num_heads=text_num_heads, dropout=dropout)  # 128 -- 128
        #  池化为固定维度： 16*16
        self.FixedPooling = FixedPooling(fixed_size=self.attention_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, user_text, user_neighbor_index, all_user_feature, edge_index):
        """
        user_neighbor_index: dict(n * 1) * num_batch
        all_user_feature: tensor(768 * 229580)
        text: tensor(768 * length) * num_bacth
        attention_graph = num_batch * dim * dim
        """
        # user_index 中心用户节点
        user_index = []
        # 遍历节点邻居
        for neighbor_index in user_neighbor_index:  # 每个用户的邻居
            user_index.append(neighbor_index[0])
        user_index = torch.tensor(user_index)  # 中心用户节点

        subgraph_loader = NeighborSampler(edge_index=edge_index, node_idx=user_index, sizes=[-1],
                                          batch_size=len(user_index))

        text, attention = self.LModel(user_text)  # torch.Size([64, 201, 128])，torch.Size([64, 201, 201]), 语言模型后得到的表示

        for _, _, adj in subgraph_loader:
            index = adj[0].to(self.device)  # torch.Size([2, 1266])
            all_user_feature = self.dropout(self.relu(self.GCN(all_user_feature, index)))  # 经过图卷积后 特征表示
        # all_user_feature = self.GCN(all_user_feature, edge_index)
        all_user_feature = self.MultiAttn(user_neighbor_index, all_user_feature)  # 注意力机制后节点的编码
        attention_graph = self.FixedPooling(attention)

        return text, all_user_feature, attention_graph


# 模型交互
class InteractLayer(nn.Module):
    def __init__(self, in_channels=768, out_channels=768):
        super(InteractLayer, self).__init__()
        self.linear_text = nn.Linear(in_channels, out_channels)  # 128 -- 128
        self.linear_graph = nn.Linear(in_channels, out_channels)  # 128 -- 128

    def forward(self, text, all_user_feature, user_neighbor_index):

        assert len(user_neighbor_index) == len(text)
        user_index = []
        for neighbor_index in user_neighbor_index:
            user_index.append(neighbor_index[0])  # 中心节点

        graph_ini = all_user_feature[user_index]  # torch.Size([64, 128]) 中心用户节点
        text_ini, text_rest = text.split([1, 200],
                                         dim=1)  # torch.Size([64, 1, 128])(interact),torch.Size([64, 200, 128])
        text_ini = text_ini.squeeze(1)  # 64,128

        text_tmp = self.linear_text(text_ini)  # torch.Size([64, 128]) 用户节点对应的文本表示
        softmax = nn.Softmax(dim=0)
        a = torch.mul(text_ini, text_tmp).sum(dim=-1).unsqueeze(-1)  # torch.Size([64, 1]) 计算文本相似度 w_hh,
        b = torch.mul(graph_ini, text_ini).sum(dim=-1).unsqueeze(-1)  # 计算文本与图相似度 torch.Size([64, 1]) w_hg
        a_b = torch.stack((a, b))  # torch.Size([2, 64, 1])
        a_b = softmax(a_b)  # 归一化 torch.Size([2, 64, 1])
        a, b = a_b.split([1, 1], dim=0)  # torch.Size([1, 64, 1])
        a, b = a.squeeze(0), b.squeeze(0)  # torch.Size([64, 1])

        text = torch.mul(a, text_ini) + torch.mul(b, graph_ini)  # 计算派生的相似性权重的帮助下交互这两种表示

        text = torch.cat((text.unsqueeze(1), text_rest), dim=1)  # 与文本其余部分连接

        graph_tmp = self.linear_graph(graph_ini)  # 线性表示
        c = torch.mul(graph_tmp, graph_ini).sum(dim=-1).unsqueeze(-1)  # 计算相似度 w_gg
        d = torch.mul(graph_ini, text_ini).sum(dim=-1).unsqueeze(-1)  # w_gh
        c_d = torch.stack((c, d))  # 连接
        c_d = softmax(c_d)  # 归一化计算权重
        c, d = c_d.split([1, 1], dim=0)
        c, d = c.squeeze(0), d.squeeze(0)
        graph = torch.mul(c, graph_ini) + torch.mul(d, text_ini)  # 图交互表示

        for i in range(len(user_index)):
            all_user_feature[user_index[i]] = graph[i]  # 更新图节点嵌入

        return text, all_user_feature


class MultiAttn(nn.Module):
    """

    """

    def __init__(self, embed_dim=768, num_heads=4):
        super(MultiAttn, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)  # 128 -- 128

    def forward(self, user_neighbor_index, all_user_feature):
        for user_index in user_neighbor_index:
            tmp_feature = all_user_feature[user_index].unsqueeze(0)  # 获取中心节点对应特征嵌入
            tmp_feature, attention_weight = self.multihead_attention(tmp_feature, tmp_feature,
                                                                     tmp_feature)  # 注意力一下 #batch个multi-attention
            all_user_feature[user_index[0]] = tmp_feature[0][0]
        return all_user_feature  # 更新中心节点嵌入


class LModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.5, activation='LeakyReLU',
                 norm_first=True, layer_norm_eps=1e-5):
        super(LModel, self).__init__()
        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,

                                                         dropout=dropout, batch_first=True)  # 128 --128
        # 激活函数选择
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        if activation == 'SELU':
            self.activation = nn.SELU()
        self.activation = nn.SELU()

        # 全连接
        self.linear1 = nn.Linear(embed_dim, embed_dim)  # 128 --128
        # dropout
        self.dropout = nn.Dropout(p=dropout)
        # 全连接
        self.linear2 = nn.Linear(embed_dim, embed_dim)  # 128 -- 128

        self.norm_first = norm_first
        # 对一小批输入应用层规范化
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, text_src):
        if self.norm_first:  # norm_first True
            text, attention_weight = self._sa_block(self.norm1(text_src))
            text = text_src + text  # text_src torch.Size([64, 201, 128]); text: torch.Size([64, 201, 128]) 残差
            text = text + self._ff_block(self.norm2(text))
        else:
            text, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + text)
            text = self.norm2(text + self._ff_block(text))
        return text, attention_weight

    def _sa_block(self, text):
        text, attention_weight = self.multihead_attention(text, text,
                                                          text)  # attention_weight: torch.Size([64, 201, 201])
        text = self.dropout1(text)
        return text, attention_weight

    def _ff_block(self, text):
        text = self.linear2(self.dropout(self.activation(self.linear1(text))))  # 提取特征 torch.Size([64, 201, 128])
        text = self.dropout2(text)
        return text


class MultiDomainFEND_Only_Attention_Model(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(MultiDomainFEND_Only_Attention_Model, self).__init__()
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        self.num_expert = 5
        self.domain_num = 4
        for i in range(self.num_expert):
            expert.append(InteractModel().cuda())
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.num_expert),
                                  nn.Softmax(dim=1)).cuda()
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim).cuda()
        self.classifier = MLP(128, mlp_dims, dropout).cuda()

    def forward(self, text, category, user_neighbor_index, num_feature, cat_feature, edge_index):

        idxs = torch.tensor([index for index in category]).view(-1, 1)
        idxs = idxs.to(device)
        # print(idxs)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        one_user_tweet = torch.mean(text, dim=1)

        gate_input = torch.cat([domain_embedding, one_user_tweet], dim=-1)

        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](text, user_neighbor_index, num_feature, cat_feature, edge_index)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))


import random
import torch

if __name__ == '__main__':
    model = MultiDomainFENDModel(768, [384], 0.2)
    tm_text = torch.rand((64, 201, 768))
    tm_text = tm_text.to(device)
    m = []
    for i in range(64):
        a = random.randint(0, 3)
        m.append(a)
    cat = torch.randint(0, 3, (64, 1))
    cat = cat.to(device)
    # cat = torch.IntTensor(m).to('cuda:0')
    # cat = torch.unsqueeze(cat, 1)
    out_put = model(tm_text, cat)
    print(out_put)
