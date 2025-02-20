import os
import sys
import math
import pprint

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
import random
import numpy as np
from easydict import EasyDict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util




separator = ">" * 30
line = "-" * 30

logger = util.get_root_logger()
def check_for_nan(grad):
    if torch.isnan(grad).any():
        raise ValueError("NaN detected in gradients")
def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    device = util.get_device(cfg)

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    for param in parallel_model.parameters():
        param.register_hook(check_for_nan)
    step = math.ceil(cfg.train.num_epoch / 10)
    best_mrr = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in train_loader:
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:].clamp(min=1e-6) / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1).clamp(min=1e-6)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        mrr,results = test(cfg, model, valid_data, filtered_data=filtered_data)
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)

    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    device = util.get_device(cfg)

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
    results={}
    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            results[metric]=score.item()
    mrr = (1 / all_ranking.float()).mean()

    return mrr.item(),results

def objective(config):
    cfg=EasyDict(config['cfg'])

    assert 'cfg.optimizer.lr' not in config

    # if config['tuning']:
    #     cfg.optimizer.lr=config['cfg.optimizer.lr']
    #
    #     cfg.model.short_cut=config['cfg.model.short_cut']
    #     cfg.model.layer_norm=config['cfg.model.layer_norm']
    #     cfg.model.dependent=config['cfg.model.dependent']
    #     cfg.model.remove_one_hop=config['cfg.model.remove_one_hop']

    args=config['args']
    vars=config['vars']
    device = config['device']
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    dataset = util.build_distinctive_dataset(cfg,device,config['opts.accuracy_threshold'],config['opts.recall_threshold'])
    cfg.model.num_relation = dataset.num_relations
    cfg.accuracy_tensor=dataset.accuracy_tensor
    cfg.recall_tensor=dataset.recall_tensor
    model = util.build_model(graphs=[config['opts.accuracy_graph'],config['opts.recall_graph'],config['opts.accuracy_graph_complement'],config['opts.recall_graph_complement']],cfg=cfg)




    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    filtered_data = None
    # if is_inductive:
    #     # for inductive setting, use only the test fact graph for filtered ranking
    #     filtered_data = None
    # else:
    #     # for transductive setting, use the whole graph for filtered ranking
    #     filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
    #     filtered_data = filtered_data.to(device)

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    mrr,results = test(cfg, model, test_data, filtered_data=filtered_data)
    return {'mrr':mrr,'metrics':results,'opts':config}
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg=util.load_config(args.config, context=vars)
    device=util.get_device(cfg)
    # working_dir = util.create_working_directory(cfg)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed + util.get_rank())
    random.seed(args.seed + util.get_rank())
    np.random.seed(args.seed + util.get_rank())



    config={
        'tuning':False,
        'opts.accuracy_threshold': args.accuracy_threshold,
        'opts.recall_threshold': args.recall_threshold,
        'opts.accuracy_graph': args.accuracy_graph,
        'opts.recall_graph': args.recall_graph,
        'opts.accuracy_graph_complement': args.accuracy_graph_complement,
        'opts.recall_graph_complement': args.recall_graph_complement,
        'cfg':cfg,
        'args':args,
        'vars':vars,
        'device':device,
    }
    results=objective(config)
    print(results)