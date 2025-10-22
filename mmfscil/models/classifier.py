import numpy as np
import torch

from mmcls.models.builder import (CLASSIFIERS, build_backbone, build_head,
                                  build_neck)
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.heads import MultiLabelClsHead
from mmcls.models.utils.augment import Augments


@CLASSIFIERS.register_module()
class ImageClassifierCIL(BaseClassifier):
    """Classifier for Incremental Learning in Image Classification.

    This classifier integrates a configurable backbone, optional neck, and head components,
    supports feature extraction at various stages, and includes capabilities for advanced
    training schemes like Mixup.

    Args:
        backbone (dict): Configuration dict for backbone.
        neck (dict, optional): Configuration dict for neck.
        head (dict, optional): Configuration dict for head.
        pretrained (str, optional): Path to the pretrained model.
        train_cfg (dict, optional): Training configurations.
        init_cfg (dict, optional): Initialization config for the classifier.
        mixup (float, optional): Alpha value for Mixup regularization. Defaults to 0.
        mixup_prob (float, optional): Probability of applying Mixup per batch. Defaults to 0.
    """

    def __init__(
        self,
        backbone,
        neck=None,
        head=None,
        pretrained=None,
        train_cfg=None,
        init_cfg=None,
        mixup: float = 0.,
        mixup_prob: float = 0.,
    ):
        super().__init__(init_cfg)

        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck:
            self.neck = build_neck(neck)

        if head:
            self.head = build_head(head)

        self.augments = None
        if train_cfg:
            augments_cfg = train_cfg.get('augments')
            if augments_cfg:
                self.augments = Augments(augments_cfg)

        self.mixup = mixup
        self.mixup_prob = mixup_prob

    def extract_feat(self, img, stage='neck'):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.
        """
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        multi_scale_features = None
        if self.backbone:
            # ResNet18 now always returns multi_scale features
            result = self.backbone(img)
            if isinstance(result, tuple) and len(result) == 2:
                x, multi_scale_features = result
            elif isinstance(result, tuple):
                # Handle other tuple formats (take the last element as main feature)
                x = result[-1]
                multi_scale_features = result[:-1] if len(result) > 1 else None
            else:
                x = result
        else:
            x = img

        if stage == 'backbone':
            if isinstance(x, tuple):
                x = x[-1]
            return x

        if self.neck:
            # Always pass multi_scale features to neck for enhanced skip connections
            if hasattr(self.neck, 'use_multi_scale_skip'):
                x = self.neck(x, multi_scale_features=multi_scale_features)
            else:
                x = self.neck(x)

        if stage == 'neck':
            return x

        if self.head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

            This method handles data augmentation, feature extraction, and loss computation for training.

            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                gt_label (Tensor): Ground truth labels. Shape (N, 1) for single label tasks,
                                   or (N, C) for multi-label tasks.

            Returns:
                dict[str, Tensor]: Dictionary of all computed loss components.
            """
        # Apply data augmentation if configured
        if self.augments:
            img, lam, gt_label, gt_label_aux = self.augments(img, gt_label)
        else:
            gt_label_aux = None
            lam = None

        x = self.extract_feat(img)
        
        aux_loss = None
        if isinstance(x, dict):
            aux_loss = x.get('aux_loss')  # MoE load balancing loss
            x = x['out']

        losses = dict()
        
        # Add MoE auxiliary loss for load balancing
        if aux_loss is not None:
            losses['aux_loss'] = aux_loss

        # make sure not mixup feat
        if gt_label_aux is not None:
            assert self.mixup == 0.
            loss = self.head.forward_train(x, gt_label_aux)
            losses['loss_main'] = losses['loss'] * lam
            losses['loss_aux'] = loss['loss'] * (1 - lam)
            del losses['loss']

        # Calculate feature norms for base / incremental classes
        indices_base = gt_label < self.head.base_classes
        if indices_base.any():
            losses['norm_main_base'] = torch.norm(x[indices_base])
        
        indices_inc = gt_label >= self.head.base_classes
        if indices_inc.any():
            losses['norm_main_inc'] = torch.norm(x[indices_inc])

        # mixup feat when mixup > 0, this cannot be with augment mixup
        if self.mixup > 0. and self.mixup_prob > 0. and np.random.random() > (
                1 - self.mixup_prob):
            x, gt_a, gt_b, lam = self.mixup_feat(x, gt_label, alpha=self.mixup)
            loss1 = self.head.forward_train(x, gt_a)
            loss2 = self.head.forward_train(x, gt_b)
            losses['loss'] = lam * loss1['loss'] + (1 - lam) * loss2['loss']
            losses['ls_mixup_1'] = loss1['loss']
            losses['ls_mixup_2'] = loss2['loss']
            losses['accuracy'] = loss1['accuracy']
        else:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)

        return losses

    def simple_test(self,
                    img,
                    gt_label,
                    return_backbone=False,
                    return_feat=False,
                    return_acc=False,
                    img_metas=None,
                    **kwargs):
        """Test without augmentation."""
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)
        if return_feat:
            assert not return_acc
            return x

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, post_process=not return_acc, **kwargs)
        if return_acc:
            res = res.argmax(dim=-1)
            return torch.eq(
                res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()
        return res

    @staticmethod
    def mixup_feat(feat, gt_labels, alpha=1.0):
        """
        Applies mixup augmentation directly to features.

        Args:
            feat (Tensor): Features to which mixup will be applied.
            gt_labels (Tensor): Ground truth labels corresponding to the features.
            alpha (float): Alpha parameter for the beta distribution.

        Returns:
            Tuple of mixed features and corresponding labels.
        """
        if alpha > 0:
            lam = alpha
        else:
            lam = 0.5
        if isinstance(feat, dict):
            feat = feat['out']
        batch_size = feat.size()[0]
        index = torch.randperm(batch_size).to(device=feat.device)

        mixed_feat = lam * feat + (1 - lam) * feat[index, :]
        gt_a, gt_b = gt_labels, gt_labels[index]

        return mixed_feat, gt_a, gt_b, lam