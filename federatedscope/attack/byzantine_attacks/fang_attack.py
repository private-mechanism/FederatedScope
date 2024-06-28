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

class Fang_adaptive_attacks():
    """
    Define the Fang_adaptive_attacks with can attack a specific aggregation rule with pertinence.
    """

    def __init__(self, model=None, device='cpu',config=None):
        self.str2defense = {'krum': KrumAggregator(model,device,config)._para_avg_with_krum,
                           'median': MedianAggregator(model,device,config)._aggre_with_median,
                           'trmean': TrimmedmeanAggregator(model,device,config)._aggre_with_trimmedmean,
                           'bulyan': BulyanAggregator(model,device,config)._aggre_with_bulyan}
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.config = config
        self.model = model
        

    def compute_lambda(self, model_re, all_updates, model_byzantine):
        """
        Calculate an upper bound of the scaling factor used in Fang_krum attack.
        """
        distances = []
        n_benign, d = all_updates.shape
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1)
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        distances[distances == 0] = 10000
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :n_benign - 2 - len(model_byzantine)], dim=1)
        min_score = torch.min(scores)
        term_1 = min_score / ((n_benign - len(model_byzantine) - 1) * torch.sqrt(torch.Tensor([d]))[0])
        max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])
        return term_1 + max_wre_dist

    def fang_krum(self, models):
        """
        Please refer to "Local model poisoning attacks to byzantine-robust federated learning"
        https://www.usenix.org/system/files/sec20-fang.pdf
        """
        all_updates = torch.stack([each_model[0][1] for each_model in models])
        model_re = torch.mean(all_updates, 0)
        deviation = torch.sign(model_re)
        model_byzantine = list()
        mal_id = {}
        index_ = []
        for index, model in enumerate(models):
            if model[1] < self.byzantine_node_num + 1:
                mal_id[model[1]] = model[0][0]
                model_byzantine.append(model[0])
                index_.append(index)
        for i in range(len(index_) - 1, -1, -1):
            models.pop(index_[i])
        agg_update = copy.deepcopy(model_re)
        lamda = self.compute_lambda(agg_update, all_updates, model_byzantine)
        threshold = 1e-5
        mal_update = []
        while lamda > threshold:
            mal_update = agg_update - lamda * deviation
            for id in mal_id:
                mal_update = self._insert_the_bn(mal_update, mal_id[id][1])
                models.append(((mal_id[id][0], mal_update), id))
            agg_num = self.config.aggregator.fang_krum_agg_num
            _, krum_candidate = self.str2defense['krum'](models, agg_num)
            for i in range(len(model_byzantine)):
                models.pop(-1)
            if krum_candidate[0] < self.byzantine_node_num + 1:
                for id in mal_id:
                    models.append(((mal_id[id][0], mal_update), id))
                return models
            else:
                mal_update = []
            lamda *= 0.5
        if not len(mal_update):
            mal_update = agg_update - lamda * deviation
        for id in mal_id:
            models.append(((mal_id[id][0], mal_update), id))
        return models

    ######## fang attack median #################################

    def fang_median(self, models):
        """
        Please refer to "Local model poisoning attacks to byzantine-robust federated learning"
        https://www.usenix.org/system/files/sec20-fang.pdf
        """
        model_byzantine = list()
        mal_id = {}
        index_ = []
        for index, model in enumerate(models):
            if model[1] < self.byzantine_node_num + 1:
                mal_id[model[1]] = model[0][0]
                model_byzantine.append(model[0])
                index_.append(index)
        for i in range(len(index_) - 1, -1, -1):
            models.pop(index_[i])
        if len(model_byzantine) == 0:
            return models
        else:
            temp = torch.stack([each_model[0][1] for each_model in models], 0)
            model_re = torch.mean(temp, 0)
            if len(model_byzantine) == 1:
                model_std = torch.zeros_like(model_re)
            else:
                model_std = torch.std(temp, 0)
            deviation = torch.sign(model_re)
            max_vector_low = model_re + 3 * model_std
            max_vector_hig = model_re + 4 * model_std
            min_vector_low = model_re - 4 * model_std
            min_vector_hig = model_re - 3 * model_std
            max_range = torch.cat((max_vector_low[:, None], max_vector_hig[:, None]), dim=1)
            min_range = torch.cat((min_vector_low[:, None], min_vector_hig[:, None]), dim=1)
            rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), len(model_byzantine)])).type(
                torch.FloatTensor)  # 【800，4】
            max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
                [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
            min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
                [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
            former = torch.stack([(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]).T * max_rand
            latter = torch.stack([(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T * min_rand
            mal_vec = (former + latter).T
            tuple_ = torch.chunk(mal_vec, len(model_byzantine), dim=0)
            i = 0
            for id in mal_id:
                fake_update = tuple_[i].reshape(model_re.shape)
                models.append(((mal_id[id][0], fake_update), id))
                i += 1
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
            reconstructed_model[key] = flatten_updates[start_idx:start_idx+len(init_model[key].view(-1))].reshape(init_model[key].shape)
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
