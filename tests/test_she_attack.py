# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class sampled_aggr_AlgoTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_sample0(self, cfg):
        backup_cfg = cfg.clone()
        import torch
        cfg.merge_from_file('scripts/attack_exp_scripts/byzantine_attacks/she_attack_convnet2_femnist.yaml')
        cfg.device = 1
        cfg.federate.client_num = 50
        # attack
        cfg.aggregator.byzantine_node_num = 5
        cfg.aggregator.BFT_args.attack = True
        cfg.aggregator.BFT_args.attack_method = 'she_krum'

        # defense
        cfg.aggregator.robust_rule = 'dynamic'
        cfg.aggregator.BFT_args.dynamic_weighted = False
        cfg.aggregator.BFT_args.dynamic_candidate=['krum']
        cfg.aggregator.BFT_args.krum_agg_num = 5
        cfg.aggregator.BFT_args.trimmedmean_excluded_ratio = 0.2
        cfg.eval.freq = 1
        cfg.outdir = 'test_attack/'
        cfg.expname = 'she_attack_femnist/'
        cfg.expname_tag = 'she_krum_krum'
        return backup_cfg
    
    def set_config_sample1(self, cfg):
        backup_cfg = cfg.clone()
        import torch
        cfg.merge_from_file('scripts/attack_exp_scripts/byzantine_attacks/she_attack_convnet2_femnist.yaml')
        cfg.device = 1
        cfg.federate.client_num = 50
        # attack
        cfg.aggregator.byzantine_node_num = 5
        cfg.aggregator.BFT_args.attack = True
        cfg.aggregator.BFT_args.attack_method = 'she_krum'

        # defense
        cfg.aggregator.robust_rule = 'dynamic'
        cfg.aggregator.BFT_args.dynamic_weighted = False
        cfg.aggregator.BFT_args.dynamic_candidate=['krum','median','bulyan','trmean']
        cfg.aggregator.BFT_args.krum_agg_num = 5
        cfg.aggregator.BFT_args.trimmedmean_excluded_ratio = 0.2
        cfg.eval.freq = 1
        cfg.outdir = 'test_attack/'
        cfg.expname = 'she_attack_femnist/'
        cfg.expname_tag = 'she_krum_dynamic'
        return backup_cfg


    def test_0_sample(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_sample0(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

    def test_1_sample(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_sample1(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

if __name__ == '__main__':
   unittest.main()
