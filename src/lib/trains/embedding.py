from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, BCELoss
from models.utils import _sigmoid
from .base_trainer import BaseTrainer


class ModleWithMlpAndLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithMlpAndLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch)
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class EmbeddingLoss(torch.nn.Module):
    def __init__(self, opt):
        super(EmbeddingLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
            RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit_rel = BCELoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, rel_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            output['rel_cls'] = _sigmoid(output['rel_cls'])
            rel_loss += self.crit_rel(output['rel_cls'], batch['rel_mask'],
                                      batch['rel_cls']) / opt.num_stacks

            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(
                        output['wh'] * batch['dense_wh_mask'],
                        batch['dense_wh'] * batch['dense_wh_mask']) /
                                mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'], batch['ind'],
                        batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                             batch['ind'],
                                             batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'],
                                          batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
            opt.off_weight * off_loss + opt.rel_weight * rel_loss
        loss_stats = {
            'loss': loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'off_loss': off_loss,
            'rel_loss': rel_loss
        }
        return loss, loss_stats


class EmbeddingTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(EmbeddingTrainer, self).__init__(opt, model, optimizer=optimizer)
        self.model_with_loss = ModleWithMlpAndLoss(model, self.loss)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'rel_loss']
        loss = EmbeddingLoss(opt)
        return loss_states, loss
