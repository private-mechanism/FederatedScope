import logging
import copy
import torch
import random
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.aggregators.krum_aggregator import KrumAggregator
from federatedscope.core.aggregators.median_aggregator import MedianAggregator
from federatedscope.core.aggregators.trimmedmean_aggregator import \
    TrimmedmeanAggregator
from federatedscope.core.aggregators.bulyan_aggregator import \
    BulyanAggregator
from federatedscope.attack.byzantine_attacks.fang_attack import \
    Fang_adaptive_attacks
from federatedscope.attack.byzantine_attacks.she_attack import \
    She_adaptive_attacks


logger = logging.getLogger(__name__)

torch.cuda.empty_cache()


class Weighted_sampled_robustAggregator(ClientsAvgAggregator):
    """
    randomly sample a robust aggregator in each round of the FL course
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Weighted_sampled_robustAggregator, self).__init__(model, device, config)
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.client_sampled_ratio = config.federate.sample_client_rate
        self.excluded_ratio = config.aggregator.BFT_args.trimmedmean_excluded_ratio
        self.candidate=config.aggregator.BFT_args.dynamic_candidate
        self.config = config
        self.str2attack = {'she_krum': She_adaptive_attacks(model,device,config).she_krum,
                           'she_median': She_adaptive_attacks(model,device,config).she_median,
                           'she_trimmedmean': She_adaptive_attacks(model,device,config).she_trimmedmean,
                           'she_bulyan': She_adaptive_attacks(model,device,config).she_krum,
                           'fang_krum': Fang_adaptive_attacks(model,device,config).fang_krum,
                           'fang_median': Fang_adaptive_attacks(model,device,config).fang_median,
                           'fang_trimmedmean': Fang_adaptive_attacks(model,device,config).fang_median,
                           'fang_bulyan': Fang_adaptive_attacks(model,device,config).fang_krum}
        self.str2defense = {'krum': KrumAggregator(model,device,config)._para_avg_with_krum,
                           'median': MedianAggregator(model,device,config)._aggre_with_median,
                           'trmean': TrimmedmeanAggregator(model,device,config)._aggre_with_trimmedmean,
                           'bulyan': BulyanAggregator(model,device,config)._aggre_with_bulyan}

        if 'krum' in self.candidate:
            assert 2 * self.byzantine_node_num + 2 < config.federate.client_num
        if 'trmean' in self.candidate:
            assert 2 * self.excluded_ratio < 1
        if 'bulyan' in self.candidate:
            assert 4 * self.byzantine_node_num + 3 <= config.federate.client_num



    def aggregate(self, agg_info):
        """
        To preform aggregation with a rule randomly sampled from a candidate set.

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]

        ## simulate the Byzantine attacks
        if  self.config.aggregator.BFT_args.attack == True:
            models = [((model[0][0], self._flatten_updates(model[0][1])), \
                model[1]) for model in models]
            attack_method = self.config.aggregator.BFT_args.attack_method
            logger.info(f'the attack {attack_method} is launching')
            models = self.str2attack[attack_method](models)
            models = [((model[0][0], self._reconstruct_updates(model[0][1])), \
                model[1]) for model in models]

        init_model = self.model.state_dict()
        
        if self.config.aggregator.BFT_args.dynamic_weighted == False:
        ## uniformly sampling from the candidate set
            self.rule_cur=random.choice(self.candidate)
            logger.info(f'the sampled rule is {self.rule_cur}')
            avg_model, _ = self.str2defense[self.rule_cur](models)
        
        else:
        ## weighted sampling from the candidate set. each weight is determined by the angle between\
        #  the local update and the global update
            global_delta = agg_info["global_delta"]
            global_delta_without_bn = self._flatten_updates_without_bn(global_delta)
            avg_model_ = copy.deepcopy(init_model)
            temp = []
            for rule in self.candidate:
                logger.info(f'weighted sampling: aggregate with the rule of {rule}')
                models_temp = copy.deepcopy(models)
                avg_model_, _ = self.str2defense[rule](models_temp)
                temp.append(copy.deepcopy(avg_model_))
            #############compute the angles and then sampling according to the prob
            prob = []
            slice_global_delta = global_delta_without_bn
            temp_ = []
            temp_list = copy.deepcopy(temp)
            for model in temp_list:
                temp_.append(self._flatten_updates_without_bn(model))
            TS = [torch.dot(tmp_delta,slice_global_delta)/(torch.linalg.norm(tmp_delta)* \
                    torch.linalg.norm(slice_global_delta)) for tmp_delta in temp_]
            for ele in TS:
                if ele < 0.1:
                    ele = 0
                else:
                    ele = ele.cpu()
                prob.append(ele)
            if np.sum(prob) == 0:
                prob = [1 for ele in TS]
            index = random.choices([i for i in range(len(prob))], weights=prob,k=1)
            logger.info(f'the sampling weights is {prob} \
                        and the sampled rule is {self.candidate[int(index[0])]}')
            avg_model = temp[int(index[0])]
        
        updated_model = copy.deepcopy(init_model)
        for key in init_model:
            updated_model[key] = init_model[key] + avg_model[key].cpu()
        torch.cuda.empty_cache()
        return updated_model




    ##### methods to transform the model update and tensor ####
    def _flatten_updates(self, model):
        model_update=[]
        init_model = self.model.state_dict()
        for key in init_model:
            model_update.append(model[key].view(-1))
        return torch.cat(model_update, dim = 0)
    
    def _flatten_updates_without_bn(self, model):
        model_update=[]
        init_model = self.model.state_dict()
        for key in init_model:
            if 'bn' not in key:
                model_update.append(model[key].view(-1))
        return torch.cat(model_update, dim = 0)

    def _reconstruct_updates(self, flatten_updates):
        start_idx = 0
        init_model = self.model.state_dict()
        reconstructed_model = copy.deepcopy(init_model)
        for key in init_model:
            reconstructed_model[key] = flatten_updates[start_idx:start_idx+ \
                len(init_model[key].view(-1))].reshape(init_model[key].shape)
            start_idx=start_idx+len(init_model[key].view(-1))
        return reconstructed_model
    
    def _extract_the_bn(self, model):
        temp_model = copy.deepcopy(self.model.state_dict())
        model = self._reconstruct_updates(model)
        bn_dict={}
        for key in temp_model:
            if 'bn' in key:
                bn_dict[key] = model[key]
        return bn_dict
    
    def _insert_the_bn(self, model_tensor, dict):
        model = self._reconstruct_updates(model_tensor)
        for key in dict:
            model[key] = dict[key]
        return self._flatten_updates(model)
