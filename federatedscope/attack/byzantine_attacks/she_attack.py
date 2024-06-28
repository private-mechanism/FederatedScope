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

logger = logging.getLogger(__name__)

class She_adaptive_attacks():
    """
    Define the She_adaptive_attacks which can attack a specific aggregation rule with pertinence.
    """

    def __init__(self,  model=None, device='cpu',config=None):
        self.str2defense = {'krum': KrumAggregator(model,device,config)._para_avg_with_krum,
                           'median': MedianAggregator(model,device,config)._aggre_with_median,
                           'trmean': TrimmedmeanAggregator(model,device,config)._aggre_with_trimmedmean,
                           'bulyan': BulyanAggregator(model,device,config)._aggre_with_bulyan}
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.config = config
        self.model = model



    def she_krum(self, models, dev_type = 'sign'):
        """
        Please refer to "Manipulating the Byzantine: Optimizing Model Poisoning \
        Attacks and Defensesfor Federated Learning"
        https://par.nsf.gov/servlets/purl/10286354
        """
        all_updates = torch.stack([each_model[0][1] for each_model in models])
        model_re = torch.mean(all_updates, 0)
        model_byzantine = list()
        mal_id = {}
        index_ = []
        for index, model in enumerate(models):
            if model[1] < self.byzantine_node_num + 1:
                mal_id[model[1]] = (model[0][0],\
                        self._extract_the_bn(model[0][1]))
                model_byzantine.append(model[0])
                index_.append(index)
        for i in range(len(index_) - 1, -1, -1):
            models.pop(index_[i])
        if len(model_byzantine) == 0:
            return models
        else:
            all_updates = torch.stack([each_model[0][1] for each_model in models])
            if dev_type == 'sign':
                deviation = torch.sign(model_re)
            elif dev_type == 'unit_vec':
                deviation = model_re / torch.norm(model_re)
            elif dev_type == 'std':
                deviation = torch.std(all_updates, 0)
            deviation = deviation
            threshold_diff = 1e-5
            lamda = 20.0
            lamda_fail = lamda
            lamda_succ = 0.
            while np.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = (model_re - lamda * deviation)
                for id in mal_id:
                    mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                    models.append(((mal_id[id][0], mal_update), id))
                models = [((model[0][0], self._reconstruct_updates(model[0][1])), \
                model[1]) for model in models]
                agg_res, mal_candidiate = self.str2defense['krum'](models)
                models = [((model[0][0], self._flatten_updates(model[0][1])), model[1]) for model in models]
                for i in range(len(model_byzantine)):
                    models.pop(-1)
                pre_num = 0
                num = 0
                for id in mal_candidiate:
                    if id < len(model_byzantine) + 1:
                        num += 1
                if self.config.aggregator.BFT_args.krum_agg_num == 1:
                    if num == 1:
                        lamda_succ = lamda
                        lamda = lamda + lamda_fail / 2
                    else:
                        lamda = lamda - lamda_fail / 2
                    lamda_fail = lamda_fail / 2
                else:
                    if num > 0 and num >= pre_num:
                        pre_num = num
                        lamda_succ = lamda
                        lamda = lamda + lamda_fail / 2
                    else:
                        lamda = lamda - lamda_fail / 2
                    lamda_fail = lamda_fail / 2
            mal_update = (model_re - lamda_succ * deviation)
            i=0
            for id in mal_id:
                mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                models.append(((mal_id[id][0], mal_update),id))
                i+=1
            logger.info(f'the model length is {len(models)}')
            #     if num == self.config.aggregator.BFT_args.krum_agg_num:
            #         lamda_succ = lamda
            #         lamda = lamda + lamda_fail / 2
            #     else:
            #         lamda = lamda - lamda_fail / 2
            #     lamda_fail = lamda_fail / 2
            # mal_update = model_re - lamda_succ * deviation
            # for id in mal_id:
            #     models.append(((mal_id[id], mal_update), id))
            return models


    def she_median(self, models, dev_type= 'sign'):
        """
        Please refer to "Manipulating the Byzantine: Optimizing Model Poisoning \
        Attacks and Defensesfor Federated Learning"
        https://par.nsf.gov/servlets/purl/10286354
        """
        all_updates = torch.stack([each_model[0][1] for each_model in models])
        model_re = torch.mean(all_updates, 0)
        threshold = 5.0
        threshold_diff = 1e-5
        model_byzantine = list()
        mal_id = {}
        index_ = []
        for index, model in enumerate(models):
            if model[1] < self.byzantine_node_num + 1:
                mal_id[model[1]] = (model[0][0],self._extract_the_bn(model[0][1]))
                model_byzantine.append(model[0])
                index_.append(index)
        for i in range(len(index_) - 1, -1, -1):
            models.pop(index_[i])

        if len(model_byzantine) == 0:
            return models
        else:
            all_updates = torch.stack([each_model[0][1] for each_model in models])
            if dev_type == 'sign':
                deviation = torch.sign(model_re)
            elif dev_type == 'unit_vec':
                deviation = model_re / torch.norm(model_re)
            elif dev_type == 'std':
                deviation = torch.std(all_updates, 0)
            lamda = threshold
            threshold_diff = threshold_diff
            prev_loss = -1
            lamda_fail = lamda
            lamda_succ = 0.
            while np.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = model_re - lamda * deviation
                for id in mal_id:
                    mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                    models.append(((mal_id[id], mal_update), id))
                mal_updates = torch.stack([each_model[0][1] for each_model in models])
                for i in range(len(model_byzantine)):
                    models.pop(-1)
                agg_grads = torch.median(mal_updates, 0)[0]
                loss = torch.norm(agg_grads - model_re)
                if prev_loss < loss:
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2
                lamda_fail = lamda_fail / 2
                prev_loss = loss
            mal_update = model_re - lamda_succ * deviation
            i = 0
            for id in mal_id:
                mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                models.append(((mal_id[id], mal_update), id))
                i += 1
            logger.info(f'the model length is {len(models)}')
            return models
        

    def she_trimmedmean(self, models, dev_type= 'sign'):
        """
        Please refer to "Manipulating the Byzantine: Optimizing Model Poisoning \
        Attacks and Defensesfor Federated Learning"
        https://par.nsf.gov/servlets/purl/10286354
        """
        all_updates = torch.stack([each_model[0][1] for each_model in models])
        model_re = torch.mean(all_updates, 0)
        threshold = 5.0
        threshold_diff = 1e-1
        model_byzantine = list()
        mal_id = dict()
        index_ = list()
        for index, model in enumerate(models):
            if model[1] < self.byzantine_node_num + 1:
                mal_id[model[1]] = (model[0][0],self._extract_the_bn(model[0][1]))
                model_byzantine.append(model[0])
                index_.append(index)
        for i in range(len(index_) - 1, -1, -1):
            models.pop(index_[i])

        if len(model_byzantine) == 0:
            return models
        else:
            all_updates = torch.stack([each_model[0][1] for each_model in models])
            if dev_type == 'sign':
                deviation = torch.sign(model_re)
            elif dev_type == 'unit_vec':
                deviation = model_re / torch.norm(model_re)
            elif dev_type == 'std':
                deviation = torch.std(all_updates, 0)
            lamda = threshold
            threshold_diff = threshold_diff
            prev_loss = -1
            lamda_fail = lamda
            lamda_succ = 0
            while np.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = model_re - lamda * deviation
                for id in mal_id:
                    mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                    models.append(((mal_id[id][0], mal_update), id))
                for i in range(len(model_byzantine)):
                    models.pop(-1)
                models = [((model[0][0], self._reconstruct_updates(model[0][1])), \
                model[1]) for model in models]
                agg_grads,_ = self.str2defense['trmean'](models)
                models = [((model[0][0], self._flatten_updates(model[0][1])), \
                model[1]) for model in models]
                agg_grads = self._flatten_updates(agg_grads)
                loss = torch.norm(agg_grads - model_re)
                if prev_loss < loss:
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2
                lamda_fail = lamda_fail / 2
                prev_loss = loss
            mal_update = (model_re - lamda_succ * deviation)
            for id in mal_id:
                mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                models.append(((mal_id[id], mal_update), id))
            return models


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
            reconstructed_model[key] = flatten_updates[start_idx:start_idx\
                +len(init_model[key].view(-1))].reshape(init_model[key].shape)
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
