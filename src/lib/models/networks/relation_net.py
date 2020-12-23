from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from ..utils import _tranpose_and_gather_feat, _gather_feat


class CombinedNet(torch.nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_classes, combined_way):
        super(CombinedNet, self).__init__()
        dim_dict = {'add': 1, 'cat': 2, 'dot': 1}

        self.combined_way = combined_way
        self.inp_dim = dim_dict[self.combined_way] * inp_dim
        self.fc = nn.Sequential(nn.Linear(self.inp_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, num_classes))
        # self.fc = torch.nn.Linear(self.inp_dim, num_classes)

    def forward(self, sub_feat, obj_feat):
        if self.combined_way == 'add':
            out = torch.add(sub_feat, obj_feat)
        elif self.combined_way == 'cat':
            out = torch.cat((sub_feat, obj_feat), dim=-1)
        elif self.combined_way == 'dot':
            out = torch.mul(sub_feat, obj_feat)
        out = self.fc(out)
        return out


class RelationNet(torch.nn.Module):
    def __init__(self, base_model, opt):
        super(RelationNet, self).__init__()
        self.num_classes = opt.num_classes_verb
        self.embedding_dims = opt.embedding_dims
        self.hidden_dims = opt.hidden_dims
        self.combined_way = opt.combined_way

        self.base_model = base_model
        self.classifier = CombinedNet(self.embedding_dims, self.hidden_dims,
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

        rel_classes = self.classifier(sub_feat, obj_feat)

        outputs['rel_cls'] = rel_classes

        return [outputs]

    def inference(self, images):
        outputs = self.base_model(images)
        return outputs

    @torch.no_grad()
    def decode(self,
               outputs,
               K_human=100,
               K_obj=100,
               K_rel=100,
               corremat=None,
               is_sub_verb=0):
        heat_obj = outputs['hm'].sigmoid_()
        wh = outputs['wh']
        embeddings = outputs['embedding']
        reg = outputs.get('reg', None)

        batch, cat_obj, height, width = heat_obj.size()
        assert batch == 1, 'batch_size must be 1'
        heat_obj = self._nms(heat_obj)
        heat_human = heat_obj[:, 0, :, :].view(batch, 1, height, width)

        # return (batch, K)
        scores_obj, inds_obj, clses_obj, ys_obj, xs_obj = self._topk(heat_obj,
                                                                     K=K_obj)
        scores_obj = scores_obj.view(batch, K_obj, 1)
        clses_obj = clses_obj.view(batch, K_obj, 1).float()

        scores_human, inds_human, clses_human, ys_human, xs_human = self._topk(
            heat_human, K=K_human)
        scores_human = scores_human.view(batch, K_human, 1)
        clses_human = clses_human.view(batch, K_human, 1).float()

        _, embedding_dims, _, _ = embeddings.size()
        embedding_human = _tranpose_and_gather_feat(embeddings, inds_human)
        embedding_obj = _tranpose_and_gather_feat(embeddings, inds_obj)

        sub = embedding_human.view(K_human, 1, embedding_dims).expand(
            K_human, K_obj, embedding_dims)
        obj = embedding_obj.expand(K_human, K_obj, embedding_dims)
        # output_rel (K_human, K_obj, verb_classes)
        output_rel = self.classifier(sub, obj).sigmoid_()
        output_rel = output_rel * scores_human.view(K_human, 1, 1) * scores_obj.view(1, K_obj, 1)

        if corremat is not None:
            # (K_obj, verb_classes)
            this_corremat = corremat[clses_obj.view(K_obj).long(), :]
        output_rel = output_rel * this_corremat

        # input (1, verb_classes, K_human, K_obj), output (1, K_rel)
        scores_rel, inds_rel, clses_rel, pos_sub, pos_obj = self._topk(
            output_rel.permute(2, 0, 1).unsqueeze(0), K_rel)
        # score_hoi = scores_human.view(
        #     1, K_rel)[:, pos_sub.view(K_rel)] * scores_rel * scores_obj.view(
        #         1, K_rel)[:, pos_obj.view(K_rel)]
        score_hoi = scores_rel

        rel_triplet = (torch.cat(
            (pos_sub.view(K_rel, 1).float(), pos_obj.view(K_rel, 1).float(),
             clses_rel.view(K_rel, 1).float(), score_hoi.view(K_rel, 1)),
            1)).cpu().numpy()

        if reg is not None:
            reg_obj = _tranpose_and_gather_feat(reg, inds_obj)
            reg_obj = reg_obj.view(batch, K_obj, 2)

            reg_human = _tranpose_and_gather_feat(reg, inds_human)
            reg_human = reg_human.view(batch, K_human, 2)

            xs_human = xs_human.view(batch, K_human, 1) + reg_human[:, :, 0:1]
            ys_human = ys_human.view(batch, K_human, 1) + reg_human[:, :, 1:2]
            xs_obj = xs_obj.view(batch, K_obj, 1) + reg_obj[:, :, 0:1]
            ys_obj = ys_obj.view(batch, K_obj, 1) + reg_obj[:, :, 1:2]

        wh_human = _tranpose_and_gather_feat(wh, inds_human)
        wh_obj = _tranpose_and_gather_feat(wh, inds_obj)
        wh_obj = wh_obj.view(batch, K_obj, 2)
        wh_human = wh_human.view(batch, K_human, 2)

        obj_bboxes = torch.cat([
            xs_obj - wh_obj[..., 0:1] / 2, ys_obj - wh_obj[..., 1:2] / 2,
            xs_obj + wh_obj[..., 0:1] / 2, ys_obj + wh_obj[..., 1:2] / 2
        ], dim=2)

        obj_detections = torch.cat([obj_bboxes, scores_obj, clses_obj], dim=2)

        human_bboxes = torch.cat([
            xs_human - wh_human[..., 0:1] / 2, ys_human -
            wh_human[..., 1:2] / 2, xs_human + wh_human[..., 0:1] / 2,
            ys_human + wh_human[..., 1:2] / 2
        ], dim=2)
        if is_sub_verb > 0:
            clses_human[:] = 0.0

        human_detections = torch.cat([human_bboxes, scores_human, clses_human], dim=2)
        return obj_detections, human_detections, rel_triplet

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(heat, (kernel, kernel),
                                        stride=1,
                                        padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        # topk_scores, topk_inds : (batch, cat, K), index in one cat
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width)
        topk_xs = (topk_inds % width)

        # topk_score, topk_ind : (batch, K)
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind // K).int()
        # topk_inds : (batch, K), index in one image
        topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                                 topk_ind).view(batch, K)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1),
                               topk_ind).view(batch, K)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1),
                               topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
