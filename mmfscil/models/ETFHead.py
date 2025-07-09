import math
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import get_dist_info

from mmcls.models.builder import HEADS, LOSSES
from mmcls.models.heads import ClsHead
from mmcls.utils import get_root_logger


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


@LOSSES.register_module()
class DRLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, reg_lambda=0.):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
        self,
        feat,
        target,
        h_norm2=None,
        m_norm2=None,
        avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2))**2) / h_norm2)

        return loss * self.loss_weight


@LOSSES.register_module()
class CombinedLoss(nn.Module):
    """Combined loss of DRLoss and CrossEntropyLoss.
    
    Args:
        dr_weight (float): Weight for DRLoss. Default: 1.0
        ce_weight (float): Weight for CrossEntropyLoss. Default: 1.0
    """
    def __init__(self, dr_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dr_loss = DRLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dr_weight = dr_weight
        self.ce_weight = ce_weight

    def forward(self, feat, target, labels=None, etf_vec=None, **kwargs):
        dr_loss = self.dr_loss(feat, target)
        
        # Cross Entropy Loss 계산
        if etf_vec is not None:
            # ETF 벡터를 사용하여 로짓 계산
            logits = torch.matmul(feat, etf_vec)  # (batch_size, num_classes)
            if labels is None:
                # 레이블이 제공되지 않은 경우, target에서 가장 큰 값의 인덱스를 사용
                labels = torch.argmax(target, dim=1)
            ce_loss = self.ce_loss(logits, labels)
        else:
            # ETF 벡터가 없는 경우 CE Loss를 0으로 설정
            ce_loss = torch.tensor(0.0, device=feat.device)
        
        total_loss = self.dr_weight * dr_loss + self.ce_weight * ce_loss
        
        # Add individual losses to kwargs for logging
        if 'log_vars' in kwargs:
            kwargs['log_vars'].update({
                'dr_loss': dr_loss.item(),
                'ce_loss': ce_loss.item()
            })
        
        return total_loss


@HEADS.register_module()
class ETFHead(ClsHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args,
                 **kwargs) -> None:
        if kwargs.get('eval_classes', None):
            self.eval_classes = kwargs.pop('eval_classes')
        else:
            self.eval_classes = num_classes
        self.base_classes = self.eval_classes

        # training settings about different length for different classes
        if kwargs.pop('with_len', False):
            self.with_len = True
        else:
            self.with_len = False

        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels

        logger = get_root_logger()
        logger.info("ETF head : evaluating {} out of {} classes.".format(
            self.eval_classes, self.num_classes))
        logger.info("ETF head : with_len : {}".format(self.with_len))

        orth_vec = generate_random_orthogonal_matrix(self.in_channels,
                                                     self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(
            torch.ones(self.num_classes, self.num_classes),
            (1 / self.num_classes))
        etf_vec = torch.mul(
            torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect

    def pre_logits(self, x):
        if isinstance(x, dict):
            x = x['out']
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor,
                      **kwargs) -> Dict:
        """Forward training data."""
        if isinstance(x, dict):
            x = x['out']
        x = self.pre_logits(x)
        if self.with_len:
            etf_vec = self.etf_vec * self.etf_rect.to(
                device=self.etf_vec.device)
            target = (etf_vec * self.produce_training_rect(
                gt_label, self.num_classes))[:, gt_label].t()
        else:
            target = self.etf_vec[:, gt_label].t()

        # CombinedLoss에 원본 레이블 전달
        if isinstance(self.compute_loss, CombinedLoss):
            losses = self.loss(x, target, labels=gt_label)
        else:
            losses = self.loss(x, target)

        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                acc = self.compute_accuracy(cls_score[:, :self.eval_classes],
                                            gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def mixup_extra_training(self, x: torch.Tensor) -> Dict:
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        assigned = torch.argmax(cls_score[:, self.eval_classes:], dim=1)
        target = self.etf_vec[:, assigned + self.eval_classes].t()
        losses = self.loss(x, target)
        return losses

    def loss(self, feat, target, **kwargs):
        losses = dict()
        # compute loss
        if self.with_len:
            if isinstance(self.compute_loss, CombinedLoss):
                loss = self.compute_loss(feat, target, etf_vec=self.etf_vec, m_norm2=torch.norm(target, p=2, dim=1), **kwargs)
            else:
                loss = self.compute_loss(feat, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            if isinstance(self.compute_loss, CombinedLoss):
                loss = self.compute_loss(feat, target, etf_vec=self.etf_vec, **kwargs)
            else:
                loss = self.compute_loss(feat, target)
        losses['loss'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False):
        if isinstance(x, dict):
            x = x['out']
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    @staticmethod
    def produce_training_rect(label: torch.Tensor, num_classes: int):
        rank, world_size = get_dist_info()
        if world_size > 0:
            recv_list = [None for _ in range(world_size)]
            dist.all_gather_object(recv_list, label.cpu())
            new_label = torch.cat(recv_list).to(device=label.device)
            label = new_label
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = torch.tensor(batch_size / uni_label_num,
                             device=label.device,
                             dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device,
                                             dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        return rect