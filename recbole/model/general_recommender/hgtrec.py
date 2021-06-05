# -*- coding: utf-8 -*-
# @Time   : 2021/6/5
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/6/5
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
HGTRec: Extension of HERec using HGT
################################################
Reference:
    Chuan Shi et al. "Heterogeneous Information Network Embedding for Recommendation." IEEE Trans. Knowl. Data Eng.
    Ziniu Hu et al. "Heterogeneous Graph Transformer." WWW 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class HGTRec(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.rating_field = config['RATING_FIELD']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.gnn_type = config['gnn_type']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.type_seq = config['type_seq']
        self.hin_schema = config['hin_schema']
        self.fields_in_same_space = config['fields_in_same_space']

        # HIN construction
        self.node_num_list = torch.LongTensor([dataset.num(_) for _ in self.type_seq])
        self.node_num = self.node_num_list.sum().item()
        cum_type_num = self.node_num_list.cumsum(0)
        self.type_start_id = {
            _: cum_type_num[i - 1].item() if i > 0 else 0
            for i, _ in enumerate(self.type_seq)
        }

        self.hin = self.hin_construction(dataset).to(config['device'])

        # define layers and loss
        self.emb = nn.Embedding(self.node_num, self.embedding_size)
        xavier_normal_initialization(self.emb)

        self.gnns = nn.ModuleList()
        for i in range(self.n_layers):
            if self.gnn_type == 'graphsage':
                from torch_geometric.nn import SAGEConv
                gnn_conv = SAGEConv(self.hidden_size, self.hidden_size)
            elif self.gnn_type == 'hgt':
                from pyHGT.conv import HGTConv
                gnn_conv = HGTConv(self.hidden_size, self.hidden_size,
                    self.num_types, self.num_relations, self.n_heads, use_RTE=False)
            self.gnns.append(gnn_conv)

        self.loss = nn.MSELoss()

    def true_type_name(self, name):
        for field_list in self.fields_in_same_space:
            if name in field_list:
                return field_list[0]
        return name

    def get_type_start_id(self, name):
        name = self.true_type_name(name)
        return self.type_start_id[name]

    @property
    def num_types(self):
        return len(self.type_seq)

    @property
    def num_relations(self):
        return 2 * len(self.hin_schema)

    def hin_construction(self, dataset):
        edge_index = []
        edge_type = []
        for i, prefix in enumerate(self.hin_schema):
            s_name, t_name = self.hin_schema[prefix]

            feat = getattr(dataset.dataset, f'{prefix}_feat')
            u_s = feat[s_name] + self.get_type_start_id(s_name)
            u_t = feat[t_name] + self.get_type_start_id(t_name)
            edge = torch.stack([torch.cat([u_s, u_t]), torch.cat([u_t, u_s])])
            cur_type = torch.cat([torch.full(u_s.shape, i*2), torch.full(u_s.shape, i*2+1)])
            edge_index.append(edge)
            edge_type.append(cur_type)

        node_type = []
        for i in range(len(self.type_seq)):
            node_type.append(torch.full((self.node_num_list[i].item(),), i))

        hin_graph = Data()
        hin_graph.x = torch.arange(0, self.node_num, dtype=torch.long)
        hin_graph.node_type = torch.cat(node_type)
        hin_graph.edge_index = torch.cat(edge_index, dim=-1)
        hin_graph.edge_type = torch.cat(edge_type)
        return hin_graph

    def forward(self, user, item):
        x = self.emb(self.hin.x)
        for i in range(self.n_layers):
            if self.gnn_type == 'graphsage':
                x = self.gnns[i](x, self.hin.edge_index)
                if i != self.n_layers - 1:
                    x = F.relu(x)
            elif self.gnn_type == 'hgt':
                x = self.gnns[i](x, self.hin.node_type, self.hin.edge_index, self.hin.edge_type, None)

        user_emb, item_emb = x[user], x[item]

        score = torch.mul(user_emb, item_emb).sum(dim=1)

        return score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.type_start_id[self.ITEM_ID]
        rating = interaction[self.rating_field]
        scores = self.forward(user, item)
        loss = self.loss(scores, rating)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.type_start_id[self.ITEM_ID]
        return self.forward(user, item)
