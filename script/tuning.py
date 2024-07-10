import os
import sys


import torch

import random
import numpy as np
import ray
from ray import tune

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util

from run import objective




if __name__ == "__main__":

    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    # working_dir = util.create_working_directory(cfg)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed + util.get_rank())
    random.seed(args.seed + util.get_rank())
    np.random.seed(args.seed + util.get_rank())
    device=util.get_device(cfg)

    search_space = {


        #
        # 'cfg.optimizer.lr': tune.choice([random.randint(1,9)*pow(10,-i) for i in range(2, 5)]),
        # 'cfg.model.short_cut': tune.choice(['yes', 'no']),
        # 'cfg.model.layer_norm': tune.choice(['yes', 'no']),
        # 'cfg.model.dependent': tune.choice(['yes', 'no']),
        # 'cfg.model.remove_one_hop': tune.choice(['yes', 'no']),
        #
        # 'opts.accuracy_threshold': 0.03,
        # 'opts.recall_threshold': 0.04,
        # 'opts.accuracy_graph': False,
        # 'opts.recall_graph': False,
        # 'opts.accuracy_graph_complement': False,
        # 'opts.recall_graph_complement': False,

        'opts.accuracy_threshold': tune.choice([0.1*i for i in range(1,9)]+[0.01*i for i in range(1,9)]+[0.001*i for i in range(1,9)]),
        'opts.recall_threshold': tune.choice([0.1*i for i in range(1,9)]+[0.01*i for i in range(1,9)]+[0.001*i for i in range(1,9)]),
        'opts.accuracy_graph': tune.choice([True, False]),
        'opts.recall_graph': tune.choice([True, False]),
        'opts.accuracy_graph_complement': tune.choice([True, False]),
        'opts.recall_graph_complement': tune.choice([True, False]),



        'cfg': cfg,
        'tuning': True,
        'args': args,
        'vars': vars,
        'device': device,
    }
    ray.init()
    results = tune.run(objective, config=search_space, resources_per_trial={"GPU": 1, "CPU":4},
                       num_samples=20)
    best_trial = results.get_best_trial("mrr", "max", "last")
    print(args)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best test MRR: {best_trial.last_result['mrr']}")
    print(best_trial.last_result['metrics'])
    print(results.get_best_config("mrr", "max", "last"))


