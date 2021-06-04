# -*- coding: utf-8 -*-
# @Time   : 2021/6/3
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2021/6/3
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

r"""
HERec
################################################
Reference:
    Chuan Shi et al. "Heterogeneous Information Network Embedding for Recommendation." IEEE Trans. Knowl. Data Eng.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class HERec(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.label = config['LABEL_FIELD']
        self.rating_field = config['RATING_FIELD']

        self.meta_path_embedding_size = config['meta_path_embedding_size']
        self.fused_user_embedding_size = config['fused_user_embedding_size']
        self.fused_item_embedding_size = config['fused_item_embedding_size']
        self.embedding_size = config['embedding_size']
        self.meta_path_embeddings_name = config['additional_feat_suffix']
        self.fusion_function = config['fusion_function']

        if self.fusion_function not in {'sl', 'pl', 'spl'}:
            raise ValueError(f'No such `fusion_function`: {self.fusion_function}!')

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.set_fused_embeddings(
            dataset, 'user', self.n_users, self.meta_path_embedding_size, self.fused_user_embedding_size
        )
        self.set_fused_embeddings(
            dataset, 'item', self.n_items, self.meta_path_embedding_size, self.fused_item_embedding_size
        )

        self.user_meta_path_embedding = nn.Embedding(self.n_users, self.fused_item_embedding_size)
        self.item_meta_path_embedding = nn.Embedding(self.n_items, self.fused_user_embedding_size)

        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.beta = nn.Parameter(torch.ones(1, dtype=torch.float))

        self.loss = nn.MSELoss()

    def set_fused_embeddings(self, dataset, name, num, old_embedding_size, new_embedding_size):
        embeddings = []
        for feat_name in self.meta_path_embeddings_name:
            if feat_name[0] == name[0]:
                feat = getattr(dataset.dataset, f'{feat_name}_feat')
                emb = torch.zeros((num, old_embedding_size), dtype=torch.float)
                # emb = feat['embedding'].mean(dim=0).repeat(num).view(num, old_embedding_size)
                emb[feat[feat_name], :] = feat['embedding']
                embeddings.append(emb)
        embeddings = torch.stack(embeddings)
        embeddings_num = len(embeddings)
        fusion_layers = nn.ModuleList([
            nn.Linear(old_embedding_size, new_embedding_size)
            for _ in range(embeddings_num)
        ])
        fusion_weight = torch.full((embeddings_num,), 1.0 / embeddings_num)
        requires_grad = self.fusion_function != 'sl'
        fusion_weight = nn.Parameter(fusion_weight, requires_grad=requires_grad)
        active_func = nn.Identity() if self.fusion_function != 'spl' else nn.Sigmoid()

        self.register_buffer(f'{name}_path_embeddings', embeddings)
        setattr(self, f'{name}_embeddings_num', embeddings_num)
        setattr(self, f'{name}_fusion_layers', fusion_layers)
        setattr(self, f'{name}_fusion_weight', fusion_weight)
        setattr(self, f'{name}_active_func', active_func)

    def get_fused_embedding(self, name, idx):
        assert name == 'user' or name == 'item'
        active_func = getattr(self, f'{name}_active_func')
        path_embeddings = getattr(self, f'{name}_path_embeddings')
        fusion_layers = getattr(self, f'{name}_fusion_layers')
        fusion_weight = getattr(self, f'{name}_fusion_weight')

        embeddings = torch.stack([active_func(f(e[idx])) for e, f in zip(path_embeddings, fusion_layers)])
        fused_embeddings = fusion_weight @ embeddings.transpose(0, 1)
        fused_embeddings = active_func(fused_embeddings)
        return fused_embeddings

    def get_fused_user_embedding(self, user):
        return self.get_fused_embedding('user', user)

    def get_fused_item_embedding(self, item):
        return self.get_fused_embedding('item', item)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        user_fused_emb = self.get_fused_user_embedding(user)
        item_fused_emb = self.get_fused_item_embedding(item)

        user_meta_path_emb = self.user_meta_path_embedding(user)
        item_meta_path_emb = self.item_meta_path_embedding(item)

        scores0 = torch.mul(user_emb, item_emb).sum(dim=1)
        scores1 = self.alpha * torch.mul(user_fused_emb, item_meta_path_emb).sum(dim=1)
        scores2 = self.beta * torch.mul(user_meta_path_emb, item_fused_emb).sum(dim=1)
        scores = scores0 + scores1 + scores2
        return scores

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.rating_field]
        scores = self.forward(user, item)
        loss = self.loss(scores, rating)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)
