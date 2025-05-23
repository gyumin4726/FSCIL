import copy
import os
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import build_optimizer, get_dist_info, save_checkpoint
from torch.nn.utils import clip_grad

from mmcls.core import CosineAnnealingCooldownLrUpdaterHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import (get_root_logger, wrap_distributed_model,
                         wrap_non_distributed_model)
from mmfscil.datasets import MemoryDataset


class Runner:
    """"simple runner for lr scheduler"""

    def __init__(self, max_iters, optimizer):
        self.max_iters = max_iters
        self.iter = 0
        self.optimizer = optimizer

    def step(self):
        self.iter += 1

    def current_lr(self):
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr


def get_test_loader_cfg(cfg, is_distributed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=is_distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    loader_cfg.update({
        'shuffle': False,
        'persistent_workers': False,
        'pin_memory': False,
        'round_up': False
    })
    # The specific dataloader settings
    test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
    return test_loader_cfg


def get_training_memory(cfg, model, logger, distributed, reduce='mean'):
    rank, world_size = get_dist_info()
    # extract feats for base cls
    train_dataset_cfg = copy.deepcopy(cfg.data.train)
    if train_dataset_cfg['type'] == 'RepeatDataset':
        train_dataset_cfg = train_dataset_cfg['dataset']
    train_dataset_cfg['pipeline'] = copy.deepcopy(cfg.data.test.pipeline)
    logger.info("The feat dataset config is : \n{}".format(train_dataset_cfg))
    train_ds = build_dataset(train_dataset_cfg)
    test_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    train_loader = build_dataloader(train_ds, **test_loader_cfg)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(train_ds))
    else:
        prog_bar = None
    memory = OrderedDict()
    for data in train_loader:
        # train_step or val_step here is not necessarily because
        # the purpose of them is handling the loss logging stuff.
        with torch.no_grad():
            result = model(return_loss=False, return_backbone=True, **data)
        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            if cls_id not in memory:
                memory[cls_id] = []
            memory[cls_id].append((img_id, result[idx].to(device='cpu')))

        if rank == 0:
            prog_bar.update(len(data['img']) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    logger.info("Feat init done with {} classes".format(len(memory)))

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(memory.keys()):
            memory_cls = memory[cls]
            recv_list = [None for _ in range(world_size)]
            # gather all result part
            dist.all_gather_object(recv_list, memory_cls)
            memory_cls = []
            for itm in recv_list:
                memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            if reduce == 'mean':
                memory[cls] = torch.mean(torch.stack(
                    list(map(lambda x: x[1], memory_cls))),
                                         dim=0)
            else:
                memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
    else:
        for cls in memory:
            memory_cls = memory[cls]
            if reduce == 'mean':
                memory[cls] = torch.mean(torch.stack(
                    list(map(lambda x: x[1], memory_cls))),
                                         dim=0)
            else:
                memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))

    logger.info("Memory done with {} classes".format(len(memory)))
    memory_tensor = []
    memory_label_tensor = []
    if reduce == 'mean':
        for cls in memory:
            memory_tensor.append(memory[cls])
            memory_label_tensor.append(cls)
        return torch.stack(memory_tensor), torch.tensor(memory_label_tensor)
    else:
        for cls in memory:
            memory_tensor.append(memory[cls])
            memory_label_tensor.extend([cls for _ in range(len(memory[cls]))])
        return torch.cat(memory_tensor), torch.tensor(memory_label_tensor)


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            for name, value in loss_value.items():
                log_vars[name] = value
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()
    return loss, log_vars


def get_test_memory(cfg, model, logger, distributed):
    rank, world_size = get_dist_info()
    # get test feat memory
    test_dataset_cfg = copy.deepcopy(cfg.data.test)
    logger.info("The test dataset config is : \n{}".format(test_dataset_cfg))
    test_ds = build_dataset(test_dataset_cfg)
    test_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    test_loader = build_dataloader(test_ds, **test_loader_cfg)
    # classes_names = test_ds.CLASSES
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_ds))
    else:
        prog_bar = None
    test_memory = OrderedDict()
    test_set = []
    test_gt_label = []
    for data in test_loader:
        # train_step or val_step here is not necessarily because
        # the purpose of them is handling the loss logging stuff.
        with torch.no_grad():
            result = model(return_loss=False, return_backbone=True, **data)
        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            if cls_id not in test_memory:
                test_memory[cls_id] = []
            test_memory[cls_id].append((img_id, result[idx].to(device='cpu')))

        if rank == 0:
            prog_bar.update(len(data['img']) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(test_memory.keys()):
            memory_cls = test_memory[cls]
            recv_list = [None for _ in range(world_size)]
            # gather all result part
            dist.all_gather_object(recv_list, memory_cls)
            memory_cls = []
            for itm in recv_list:
                memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            test_memory[cls] = torch.stack(
                list(map(lambda x: x[1], memory_cls)))
            test_set.append(test_memory[cls])
            test_gt_label.append(
                torch.ones((len(test_memory[cls]), ), dtype=torch.int) * cls)
    else:
        for cls in test_memory:
            memory_cls = test_memory[cls]
            test_memory[cls] = torch.stack(
                list(map(lambda x: x[1], memory_cls)))
            test_set.append(test_memory[cls])
            test_gt_label.append(
                torch.ones((len(test_memory[cls]), ), dtype=torch.int) * cls)
    test_set = torch.cat(test_set, dim=0)
    test_gt_label = torch.cat(test_gt_label, dim=0)
    logger.info("Test memory done with {} classes".format(len(test_memory)))
    return test_set, test_gt_label


def get_inc_memory(cfg, model, logger, distributed, inc_start, inc_end):
    rank, world_size = get_dist_info()
    # get incremental feat memory
    inc_dataset_cfg = copy.deepcopy(cfg.data.train)
    if inc_dataset_cfg['type'] == 'RepeatDataset':
        inc_dataset_cfg = inc_dataset_cfg['dataset']
    inc_dataset_cfg['pipeline'] = copy.deepcopy(cfg.data.test.pipeline)
    inc_dataset_cfg.update({'few_cls': tuple(range(inc_start, inc_end))})
    logger.info(
        "The incremental dataset config is : \n{}".format(inc_dataset_cfg))
    inc_ds = build_dataset(inc_dataset_cfg)
    inc_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    inc_loader = build_dataloader(inc_ds, **inc_loader_cfg)
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(inc_ds))
    else:
        prog_bar = None
    inc_memory = OrderedDict()
    for cls_id in range(inc_start, inc_end):
        inc_memory[cls_id] = []
    inc_set = []
    inc_gt_label = []
    for data in inc_loader:
        # train_step or val_step here is not necessarily because
        # the purpose of them is handling the loss logging stuff.
        with torch.no_grad():
            result = model(return_loss=False, return_backbone=True, **data)
        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            inc_memory[cls_id].append((img_id, result[idx].to(device='cpu')))

        if rank == 0:
            prog_bar.update(len(data['img']) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(inc_memory.keys()):
            memory_cls = inc_memory[cls]
            recv_list = [None for _ in range(world_size)]
            # gather all result part
            dist.all_gather_object(recv_list, memory_cls)
            memory_cls = []
            for itm in recv_list:
                memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            inc_memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
            inc_set.append(inc_memory[cls])
            inc_gt_label.append(
                torch.ones((len(inc_memory[cls]), ), dtype=torch.int) * cls)
    else:
        for cls in inc_memory:
            memory_cls = inc_memory[cls]
            inc_memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
            inc_set.append(inc_memory[cls])
            inc_gt_label.append(
                torch.ones((len(inc_memory[cls]), ), dtype=torch.int) * cls)
    inc_set = torch.cat(inc_set, dim=0)
    inc_gt_label = torch.cat(inc_gt_label, dim=0)
    logger.info("Incremental memory done with {} classes".format(
        len(torch.unique(inc_gt_label))))
    return inc_set, inc_gt_label


def test_session(cfg,
                 model,
                 distributed,
                 test_feat: torch.Tensor,
                 test_label: torch.Tensor,
                 logger,
                 session_idx: int,
                 inc_start: int,
                 inc_end: int,
                 base_num: int,
                 mode='test',
                 save_results=False):
    if mode == 'train':
        return 0
    rank, world_size = get_dist_info()
    model.eval()
    if not save_results:
        logger.info("Evaluating session {}, from {} to {}.".format(
            session_idx, inc_start, inc_end))
    test_set_memory = MemoryDataset(
        feats=test_feat[torch.logical_and(torch.ge(test_label, inc_start),
                                          torch.less(test_label, inc_end))],
        labels=test_label[torch.logical_and(torch.ge(test_label, inc_start),
                                            torch.less(test_label, inc_end))])
    test_loader_memory = build_dataloader(
        test_set_memory,
        samples_per_gpu=256,
        workers_per_gpu=8,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.get('seed'),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        round_up=False,
    )
    result_list = []
    label_list = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_memory))
    else:
        prog_bar = None

    for data in test_loader_memory:
        with torch.no_grad():
            result = model(return_loss=False,
                           return_acc=True,
                           img=data['feat'],
                           gt_label=data['gt_label'])
        result_list.extend(result)
        label_list.extend(data['gt_label'].tolist())
        if rank == 0:
            prog_bar.update(len(result) * world_size)
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_memory))
    else:
        prog_bar = None

    # To circumvent MMCV bug
    if rank == 0:
        print()
    if distributed:
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, result_list)
        results = []
        for machine in recv_list:
            results.extend(machine)
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, label_list)
        labels = []
        for machine in recv_list:
            labels.extend(machine)
    else:
        results = result_list
        labels = label_list
    assert len(results) == len(test_set_memory)
    assert len(results) == len(labels)
    results = torch.tensor(results)
    labels = torch.tensor(labels)
    acc = torch.mean(results).item() * 100.
    acc_b = torch.mean(results[labels < base_num]).item() * 100.
    acc_i = torch.mean(results[labels >= base_num]).item() * 100.
    acc_i_new = torch.mean(
        results[labels >= inc_end - cfg.inc_step]).item() * 100.
    acc_i_old = torch.mean(results[torch.logical_and(
        torch.less(labels, inc_end - cfg.inc_step), torch.ge(
            labels, base_num))]).item() * 100.
    
    if save_results:
        return {
            'session': session_idx,
            'acc': acc,
            'acc_base': acc_b,
            'acc_inc': acc_i,
            'acc_incremental_old': acc_i_old,
            'acc_incremental_new': acc_i_new
        }
    else:
        logger.info(
            "[{:02d}]Evaluation results : acc : {:.2f} ; acc_base : {:.2f} ; acc_inc : {:.2f}"
            .format(session_idx, acc, acc_b, acc_i))
        logger.info(
            "[{:02d}]Evaluation results : acc_incremental_old : {:.2f} ; acc_incremental_new : {:.2f}"
            .format(session_idx, acc_i_old, acc_i_new))

    return acc


def test_session_feat(
        cfg,
        model,
        distributed,
        test_feat: torch.Tensor,
        test_label: torch.Tensor,
        cls_feat: torch.Tensor,  # For feat compare
        logger,
        session_idx: int,
        inc_start: int,
        inc_end: int,
        base_num: int):
    logger.info("[Feat Evaluator] Extracting cls feat".format(
        session_idx, inc_start, inc_end))
    with torch.no_grad():
        cls_feat_after_neck = model(return_loss=False,
                                    return_feat=True,
                                    img=cls_feat,
                                    gt_label=None)
        cls_feat_after_neck = model.module.head.pre_logits(cls_feat_after_neck)
    logger.info("[Feat Evaluator] length of cls vector : {}.".format(
        len(cls_feat_after_neck)))
    cls_feat_after_neck_min = cls_feat_after_neck.detach().clone()
    cls_feat_after_neck_max = cls_feat_after_neck.detach().clone()
    dist.all_reduce(cls_feat_after_neck_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(cls_feat_after_neck_max, op=dist.ReduceOp.MAX)
    assert torch.allclose(cls_feat_after_neck_min, cls_feat_after_neck_max)
    rank, world_size = get_dist_info()
    model.eval()
    logger.info("[Feat Evaluator]Evaluating session {}, from {} to {}.".format(
        session_idx, inc_start, inc_end))
    test_set_memory = MemoryDataset(
        feats=test_feat[torch.logical_and(torch.ge(test_label, inc_start),
                                          torch.less(test_label, inc_end))],
        labels=test_label[torch.logical_and(torch.ge(test_label, inc_start),
                                            torch.less(test_label, inc_end))])
    test_loader_memory = build_dataloader(
        test_set_memory,
        samples_per_gpu=256,
        workers_per_gpu=8,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.get('seed'),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        round_up=False,
    )

    result_list = []
    label_list = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_memory))
    else:
        prog_bar = None
    for data in test_loader_memory:
        with torch.no_grad():
            result = model(return_loss=False,
                           return_feat=True,
                           img=data['feat'],
                           gt_label=None)
            pre_logits = model.module.head.pre_logits(result)
            pre_label = (pre_logits @ cls_feat_after_neck.t()).argmax(dim=-1)
            acc = torch.eq(pre_label, data['gt_label'].to(device=pre_label.device)).\
                to(dtype=torch.float32).cpu().numpy().tolist()
        result_list.extend(acc)
        label_list.extend(data['gt_label'].tolist())
        if rank == 0:
            prog_bar.update(len(result) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    if distributed:
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, result_list)
        results = []
        for machine in recv_list:
            results.extend(machine)

        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, label_list)
        labels = []
        for machine in recv_list:
            labels.extend(machine)
    else:
        results = result_list
        labels = label_list
    assert len(results) == len(test_set_memory)
    assert len(results) == len(labels)
    results = torch.tensor(results)
    labels = torch.tensor(labels)
    acc = torch.mean(results).item() * 100.
    acc_b = torch.mean(results[labels < base_num]).item() * 100.
    acc_i = torch.mean(results[labels >= base_num]).item() * 100.
    acc_i_new = torch.mean(
        results[labels >= inc_end - cfg.inc_step]).item() * 100.
    acc_i_old = torch.mean(results[torch.logical_and(
        torch.less(labels, inc_end - cfg.inc_step), torch.ge(
            labels, base_num))]).item() * 100.
    logger.info(
        "[{:02d}]Evaluation results : acc : {:.2f} ; acc_base : {:.2f} ; acc_inc : {:.2f}"
        .format(session_idx, acc, acc_b, acc_i))
    logger.info(
        "[{:02d}]Evaluation results : acc_incremental_old : {:.2f} ; acc_incremental_new : {:.2f}"
        .format(session_idx, acc_i_old, acc_i_new))
    return acc


def fscil(model,
          cfg,
          distributed=False,
          validate=False,
          timestamp=None,
          meta=None):
    inc_start = cfg.inc_start
    inc_end = cfg.inc_end
    inc_step = cfg.inc_step
    logger = get_root_logger()
    rank, world_size = get_dist_info()
    # put inference model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model_inf = wrap_distributed_model(
            copy.deepcopy(model),
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model_inf = wrap_non_distributed_model(copy.deepcopy(model),
                                               cfg.device,
                                               device_ids=cfg.gpu_ids)
    model_inf = model_inf.eval()
    proto_memory, proto_memory_label = get_training_memory(
        cfg, model_inf, logger, distributed)
    test_feat, test_label = get_test_memory(cfg, model_inf, logger,
                                            distributed)
    inc_feat, inc_label = get_inc_memory(cfg, model_inf, logger, distributed,
                                         inc_start, inc_end)

    # Now start to finetune
    model_finetune = copy.deepcopy(model)
    model_finetune.backbone = None
    if False:  #model.neck.detach_residual:
        model_finetune.neck.mlp_proj = None
        model_finetune.neck.pos_embed = None
        model_finetune.neck.block = None
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model_finetune = wrap_distributed_model(
            model_finetune,
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model_finetune = wrap_non_distributed_model(model_finetune,
                                                    cfg.device,
                                                    device_ids=cfg.gpu_ids)

    # acc list for print
    acc_list = []
    acc_list_train = []
    session_results = []  # 세션별 결과를 저장할 리스트
    
    # 첫 번째 세션 평가
    result = test_session(cfg, model_finetune, distributed, test_feat, test_label,
                       logger, 1, 0, inc_start, inc_start, save_results=True)
    session_results.append(result)
    acc_list.append(result['acc'])
    
    acc_train = test_session(cfg,
                             model_finetune,
                             distributed,
                             proto_memory,
                             proto_memory_label,
                             logger,
                             1,
                             0,
                             inc_start,
                             inc_start,
                             mode='train')
    acc_list_train.append(acc_train)
    
    logger.info("Start to execute the incremental sessions.")
    logger.info(f"Prototypes features shape: {proto_memory.shape}.")
    logger.info(f"Incremental features shape: {inc_feat.shape}.")
    # model_finetune.module.head.etf_rect[:, :inc_start] = 1.
    save_checkpoint(model_finetune,
                    os.path.join(cfg.work_dir, 'session_{}.pth'.format(0)))
    # class_weight = model_finetune.module.head.compute_loss.class_weight
    for i in range((inc_end - inc_start) // inc_step):
        label_start = inc_start + i * inc_step
        label_end = inc_start + (i + 1) * inc_step
        # model_finetune.module.head.etf_rect[:, inc_start:label_start] = 1.
        # model_finetune.module.head.compute_loss.class_weight = class_weight[:label_end]
        # logger.info("etf_rect : {}".format(model_finetune.module.head.etf_rect.tolist()))
        dash_line = '-' * 60
        logger.info("Starting session : {} {}".format(i + 2, dash_line))
        logger.info("Newly added classes are from {} to {}.".format(
            label_start, label_end))
        model_finetune.module.head.eval_classes = label_end
        logger.info("Model now can classify {} classes".format(
            model_finetune.module.head.eval_classes))
        # Start to train model
        model_finetune.train()
        # steps of finetune
        num_steps = cfg.step_list[i]
        logger.info("{} steps".format(num_steps))
        if num_steps > 0:
            if cfg.mean_cur_feat:
                logger.info(
                    "Extracting all mean neck feats from {} to {}".format(
                        inc_start, label_end))
                logger.info("Copy {} duplications.".format(cfg.copy_list[i]))
                mean_feat = []
                mean_label = []
                for idx in range(inc_start, label_end):
                    mean_feat.append(inc_feat[inc_label == idx].mean(
                        dim=0, keepdim=True))
                    mean_label.append(inc_label[inc_label == idx][0:1])
                cur_session_feats = torch.cat(mean_feat).repeat(
                    cfg.copy_list[i], 1, 1, 1)
                cur_session_labels = torch.cat(mean_label).repeat(
                    cfg.copy_list[i])
            elif cfg.mean_neck_feat:
                logger.info("Extracting mean neck feat from {} to {}".format(
                    inc_start, label_start))
                logger.info("Copy {} duplications.".format(cfg.copy_list[i]))
                cur_session_feats = inc_feat[torch.logical_and(
                    torch.ge(inc_label, label_start),
                    torch.less(inc_label, label_end))]
                cur_session_labels = inc_label[torch.logical_and(
                    torch.ge(inc_label, label_start),
                    torch.less(inc_label, label_end))]

                mean_feat = []
                mean_label = []
                for idx in range(inc_start, label_start):
                    mean_feat.append(inc_feat[inc_label == idx].mean(
                        dim=0, keepdim=True))
                    mean_label.append(inc_label[inc_label == idx][0:1])
                if label_start > inc_start:
                    cur_session_feats = torch.cat([
                        cur_session_feats,
                        torch.cat(mean_feat).repeat(cfg.copy_list[i], 1, 1, 1)
                    ])
                    cur_session_labels = torch.cat([
                        cur_session_labels,
                        torch.cat(mean_label).repeat(cfg.copy_list[i])
                    ])
                    # cur_session_feats = torch.cat([cur_session_feats, torch.cat(mean_feat)])
                    # cur_session_labels = torch.cat([cur_session_labels, torch.cat(mean_label)])
            else:
                logger.info("Extracting feats from {} to {}".format(
                    inc_start, label_start))
                cur_session_feats = inc_feat[torch.logical_and(
                    torch.ge(inc_label, inc_start),
                    torch.less(inc_label, label_end))]
                cur_session_labels = inc_label[torch.logical_and(
                    torch.ge(inc_label, inc_start),
                    torch.less(inc_label, label_end))]
            cur_session_feats = torch.cat([
                cur_session_feats,
                proto_memory.repeat(cfg.base_copy_list[i], 1, 1, 1)
            ],
                                          dim=0)
            cur_session_labels = torch.cat([
                cur_session_labels,
                proto_memory_label.repeat(cfg.base_copy_list[i])
            ],
                                           dim=0)
            cur_dataset = MemoryDataset(feats=cur_session_feats,
                                        labels=cur_session_labels)
            logger.info("Copy base classes {} duplications.".format(
                cfg.base_copy_list[i]))
            logger.info("Session : {} ; The dataset has {} samples.".format(
                i + 2, len(cur_dataset)))
            logger.info("Labels : {}".format(cur_session_labels.tolist()))
            cur_session_loader = build_dataloader(
                cur_dataset,
                samples_per_gpu=8,
                workers_per_gpu=8,
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.get('seed'),
                shuffle=True,
                persistent_workers=False,
                pin_memory=False,
                round_up=False,
                drop_last=True,
            )
            if isinstance(cfg.finetune_lr, float):
                cfg.finetune_lr = [cfg.finetune_lr] * len(cfg.copy_list)
            logger.info("Finetune lr is : {}".format(cfg.finetune_lr))

            optimizer = build_optimizer(model=model_finetune,
                                        cfg=dict(
                                            type='SGD',
                                            lr=cfg.finetune_lr[i],
                                            momentum=0.9,
                                            weight_decay=0.0005,
                                            paramwise_cfg=cfg.optimizer.get(
                                                'paramwise_cfg', None)))
            runner = Runner(num_steps, optimizer)
            lr_scheduler = CosineAnnealingCooldownLrUpdaterHook(
                min_lr=None,
                min_lr_ratio=1.e-2,
                cool_down_ratio=0.1,
                cool_down_time=5,
                by_epoch=False,
                # warmup
                warmup='linear',
                warmup_iters=5,
                warmup_ratio=0.1,
                warmup_by_epoch=False)
            lr_scheduler.before_run(runner)
            cur_session_loader_iter = iter(cur_session_loader)
            for idx in range(num_steps):
                runner.step()
                lr_scheduler.before_train_iter(runner)
                try:
                    data = next(cur_session_loader_iter)
                except StopIteration:
                    cur_session_loader_iter = iter(cur_session_loader)
                    if distributed:
                        # similar to DistSamplerSeedHook
                        cur_session_loader.sampler.set_epoch(idx + 1)
                    data = next(cur_session_loader_iter)
                optimizer.zero_grad()
                losses = model_finetune(return_loss=True,
                                        img=data['feat'],
                                        gt_label=data['gt_label'])
                # losses['loss'].backward()
                loss, log_vars = parse_losses(losses)
                try:
                    loss.backward()
                except:
                    logger.info(f'Backward failed, skip!')
                if cfg.grad_clip:
                    params = model_finetune.module.parameters()
                    params = list(
                        filter(
                            lambda p: p.requires_grad and p.grad is not None,
                            params))
                    if len(params) > 0:
                        max_norm = clip_grad.clip_grad_norm_(
                            params, max_norm=cfg.grad_clip)
                        logger.info("max norm : {}".format(max_norm.item()))
                optimizer.step()
                if rank == 0:
                    logger.info(
                        "[{:03d}/{:03d}] Training session : {} ; lr : {} ; loss : {} ; acc@1 : {}"
                        .format(idx + 1, num_steps, i + 2,
                                runner.current_lr()[-1], log_vars['loss'],
                                losses['accuracy']['top-1'].item()))
                    info = ""
                    for key in log_vars:
                        info = info + f"| {key}={log_vars[key]} |"
                    if info != "":
                        logger.info(info)
            # is False
            if cfg.feat_test:
                cls_feat = []
                for idx in range(0, label_end):
                    cls_feat.append(
                        cur_session_feats[cur_session_labels == idx].mean(
                            dim=0, keepdim=True))
                cls_feat = torch.cat(cls_feat)
                result = test_session_feat(cfg,
                                        model_finetune,
                                        distributed,
                                        test_feat,
                                        test_label,
                                        cls_feat,
                                        logger,
                                        i + 2,
                                        0,
                                        label_end,
                                        inc_start,
                                        mode='test')
                acc = result
            else:
                result = test_session(cfg, model_finetune, distributed, test_feat,
                                   test_label, logger, i + 2, 0, label_end,
                                   inc_start, save_results=True)
                session_results.append(result)
                acc = result['acc']
                
                acc_train = test_session(cfg,
                                         model_finetune,
                                         distributed,
                                         cur_session_feats,
                                         cur_session_labels,
                                         logger,
                                         i + 2,
                                         0,
                                         label_end,
                                         inc_start,
                                         mode='train')
            acc_list.append(acc)
            acc_list_train.append(acc_train)
            save_checkpoint(
                model_finetune,
                os.path.join(cfg.work_dir, 'session_{}.pth'.format(i + 1)))
    
    # 모든 세션의 결과를 한 번에 출력
    logger.info("\n=== Final Evaluation Results for All Sessions ===")
    for result in session_results:
        logger.info(
            "[{:02d}]Evaluation results : acc : {:.2f} ; acc_base : {:.2f} ; acc_inc : {:.2f}"
            .format(result['session'], result['acc'], result['acc_base'], result['acc_inc']))
        logger.info(
            "[{:02d}]Evaluation results : acc_incremental_old : {:.2f} ; acc_incremental_new : {:.2f}"
            .format(result['session'], result['acc_incremental_old'], result['acc_incremental_new']))
    
    # 전체 정확도 리스트 출력
    acc_str = ""
    for acc in acc_list:
        acc_str += "{:.2f} ".format(acc)
    logger.info("\n=== Accuracy Progression Across Sessions ===")
    logger.info(acc_str)
