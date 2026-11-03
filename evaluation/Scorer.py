from torch.utils.data import DataLoader
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List
import sys
import itertools
from util_img import *
class Metric:
    """ Tuple Metric """
    def __init__(self, verbose=False, match_mode='normal'):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.
        self.tn = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    def __repr__(self) -> str:
        return f"tp: {self.tp}, gold: {self.gold_num}, pred: {self.pred_num}"

    
    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0
    
    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)
            > 0
        ):
            return round(
                (self.get_tp(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                ),
                4,
            )
        return 0.
    
    def f_score(self, class_name=None):
        # import pdb 
        # pdb.set_trace()
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0
    
    def micro_avg_f_score(self):
        return self.f_score(None)

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        # import pdb
        # pdb.set_trace()
        # print(self.get_classes())
        tp = self.tp
        fn = self.fn
        fp = self.fp
        tn = self.tn
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = round(self.safe_div(tp, tp+fp), 4), round(self.safe_div(tp, tp+fn), 4)
        f1 = round(self.safe_div(2*p*r, p+r), 4)
        return p, r, f1
                
    
    
    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            # if len(gold_list) > 0 and len(pred_list) > 0:
            #     # guarantee length same
            #     # if len(gold_list[0]) != len(pred_list[0]):
            #     print(gold_list,pred_list)
            #     assert len(gold_list[0]) == len(pred_list[0])
            # print(gold_list,pred_list)
            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    dup_gold_list.remove(pred)
                else:
                    # for gold in dup_gold_list:
                    #     if pred[1] in gold[1] and gold[0] != "O":
                    #         dup_gold_list.remove(gold)
                    self.fp += 1
                            # break
                        
            # for gold in dup_gold_list:
            #     self._fps[gold[0]] += 1
            #     self.fp += 1
                    
               
            dup_gold_list = deepcopy(gold_list)
            for gold in dup_gold_list:
                if gold not in pred_list:
                    self.fn+=1
                else:
                    self.tn+=1

    def count_instance_iou(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            # if len(gold_list) > 0 and len(pred_list) > 0:
            #     # guarantee length same
            #     # if len(gold_list[0]) != len(pred_list[0]):
            #     print(gold_list,pred_list)
            #     assert len(gold_list[0]) == len(pred_list[0])
            # print(gold_list,pred_list)
            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                for dup_gold in dup_gold_list:
                    if pred['role']==dup_gold['role'] and pred['event_type']==dup_gold['event_type'] and isCorrect(dup_gold['localization'], pred['localization']):
                        self.tp += 1
                        dup_gold_list.remove(dup_gold)
                        break
                    else:
                        # for gold in dup_gold_list:
                        #     if pred[1] in gold[1] and gold[0] != "O":
                        #         dup_gold_list.remove(gold)
                        if dup_gold == dup_gold_list[-1]:
                            self.fp += 1
                            # break
                        
            # for gold in dup_gold_list:
            #     self._fps[gold[0]] += 1
            #     self.fp += 1
                    
               
            dup_gold_list = deepcopy(gold_list)
            for gold in dup_gold_list:
                for pred_ in pred_list:
                    if pred_['role']==gold['role'] and pred_['event_type']==gold['event_type'] and isCorrect(gold['localization'], pred_['localization']):
                        self.tn+=1
                        break
                    else:
                        if pred_ == pred_list[-1]:
                            self.fn+=1

    def count_instance_txt(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            
            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                for dup_gold in dup_gold_list:
                    if pred['role']==dup_gold['role'] and pred['event_type']==dup_gold['event_type'] and dup_gold['text'].lower() == pred['text'].lower():
                        self.tp += 1
                        dup_gold_list.remove(dup_gold)
                        break
                    else:
                       
                        if dup_gold == dup_gold_list[-1]:
                            self.fp += 1
                        
            dup_gold_list = deepcopy(gold_list)
            for gold in dup_gold_list:
                for pred_ in pred_list:
                    if pred_['role']==gold['role'] and pred_['event_type']==gold['event_type'] and dup_gold['text'].lower() == pred['text'].lower():
                        self.tn+=1
                        break
                    else:
                        if pred_ == pred_list[-1]:
                            self.fn+=1
    
    def equal_to(self, dict1, dict2):
        for key in dict1: 
            if dict1[key] not in dict2[key]:
                return False
        return True
    
    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        # all_classes = ["organization", "miscellaneous", "person", "location"]
        # all_classes = ["part of", "contain", "present in", "none", "held on", "member of", "peer", "place of residence", "locate at", "alternate names", "neighbor", "subsidiary", "awarded", "couple", "parent", "nationality", "place of birth", "charges", "siblings", "religion", "race"]

        return all_classes
    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)
