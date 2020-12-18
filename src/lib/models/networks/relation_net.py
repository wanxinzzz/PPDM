from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from ..utils import _tranpose_and_gather_feat


class CombinedNet(torch.nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_classes, combined_way):
        super(CombinedNet, self).__init__()
        dim_dict = {'add': 1, 'cat': 2, 'dot': 1}

        self.combined_way = combined_way
        self.inp_dim = dim_dict[self.combined_way] * inp_dim
        self.classifier = nn.Sequential(nn.Linear(self.inp_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, num_classes))
        # self.classifier = torch.nn.Linear(self.inp_dim, num_classes)

    def forward(self, sub_feat, obj_feat):
        if self.combined_way == 'add':
            out = torch.add(sub_feat, obj_feat)
        elif self.combined_way == 'cat':
            out = torch.cat((sub_feat, obj_feat), dim=-1)
        elif self.combined_way == 'dot':
            out = torch.mul(sub_feat, obj_feat)
        out = self.classifier(out)
        return out


class RelationNet(torch.nn.Module):
    def __init__(self, base_model, opt):
        super(RelationNet, self).__init__()
        self.num_classes = opt.num_classes_verb
        self.embedding_dims = opt.embedding_dims
        self.hidden_dims = opt.hidden_dims
        self.combined_way = opt.combined_way

        self.base_model = base_model
        self.mlp = CombinedNet(self.embedding_dims, self.hidden_dims,
                               self.num_classes, self.combined_way)

    def forward(self, batch):
        outputs = self.base_model(batch['input'])
        outputs = self.get_relation(outputs, batch)

        return outputs

    def get_relation(self, outputs, batch):
        outputs = outputs[-1]
        embeddings = outputs['embedding']
        sub_ind = batch['sub_ind']
        obj_ind = batch['obj_ind']

        sub_feat = _tranpose_and_gather_feat(embeddings, sub_ind)
        obj_feat = _tranpose_and_gather_feat(embeddings, obj_ind)

        rel_classes = self.mlp(sub_feat, obj_feat)

        outputs['rel_cls'] = rel_classes

        return [outputs]
