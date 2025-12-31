"""
This is the code for paper Companion rule list
We make small modification to unify the API with other models
"""
import time
import numpy as np
import pandas as pd 
import random
import operator
import math
from bitarray import bitarray
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")

#from rule_mining import generate_rulespace #uncomment if you want to do rule mining (DONE BY ZIBA)

@dataclass
class Coverage:
    """ Store results on the Coverage of a RuleList """
    rule_cover: List[bitarray] = field(default_factory=list) # Covered by a rule up to the mth
    rule_catch: List[bitarray] = field(default_factory=list) # Catch by the mth list
    bbox_cover: List[bitarray] = field(default_factory=list) # Not covered by a rule up to m
    A: List[float] = field(default_factory=list)      # Accuracy of RuleList up to m
    choose: List[int] = field(default_factory=list)   # Decision at list m



def my_debug(msg):
    print("====================================> {}".format(msg))



class CRL(object):
    
    def __init__(self, bbox, min_support=0.05, max_card=2, alpha=0.001):
        """Hybrid Rule List/Black-box based classifier.

        This class implements an Hybrid interpretable/black-box model.
        It considers that the black-box model has already been fitted and then fits a rulelist on top of
        it in order to gain free transparency. At inference, instances covered by the rules are labeled
        according to those rules while instances not covered are send to the black box for prediction.

        Attributes
        ----------

        bbox: already fitted classifier

        min_support: minimal support of the rules

        max_card: maximal cardinality of the rules

        alpha: regularizaition parameter

        References
        ----------
        Danqing Pan, Tong Wang, Satoshi Hara. Interpretable Companions for Black-Box Models. 
        In Proceedings of the Twenty Third International Conference on Artificial Intelligence 
        and Statistics, PMLR 108:2444-2454, 2020.
        """
        self.bbox = bbox
        self.min_support = min_support
        self.max_card = max_card
        assert alpha < 1, "The parameter alpha must be in the interval [0, 1)"
        self.alpha = alpha

        # List of bitarrays indicating whether or not rule i covers example j
        self.cover_sets = []



    def __propose_rule(self, rule_sequence):
        # This function propose a new rule based on the previous rule
        
        premined = list(range(self.n_rules))

        premined_rules = premined.copy()
        rule_seq = rule_sequence.copy()
        
        # No rule reuse
        for i in range(len(rule_seq)):
            premined_rules.remove(rule_seq[i])
        
        rand = random.random()
        
        # We use 4 operations to generate a new rule
        
        if (rand < 0.25):
            #print('add')
            if len(premined_rules)>1:
                # randomly choose a rule in premined rules
                rule_to_add = random.sample(premined_rules,1)[0]
                # insert to a random position in the list
                rule_seq.insert(random.randint(0,len(rule_seq)), rule_to_add)
            
        elif (rand < 0.5):
            #print('remove')
            if len(rule_seq)>1: # at least have 2 rules in the list
                # randomly choose a rule from the list
                rule_to_remove = random.sample(rule_seq,1)[0]
                # remove it
                rule_seq.remove(rule_to_remove)

        elif (rand < 0.75):
            #print('swap')
            if len(rule_seq)>1: # at least have 2 rules in the list
                # randomly choose 2 rules in the list
                swap_num = random.sample(list(range(len(rule_seq))),2)
                # swap them
                rule_seq[swap_num[0]],rule_seq[swap_num[1]] = rule_seq[swap_num[1]],rule_seq[swap_num[0]]

        else:
            #print('replace')
            if (len(rule_seq)>0 and len(premined_rules)>1):
                # randomly choose a rule in the list
                replace_num = random.sample(list(range(len(rule_seq))),1)[0]
                # randomly choose a premined rule
                rule_selected = random.sample(premined_rules,1)[0]
                # replace the rule in the list
                rule_seq[replace_num] = rule_selected
                
        return rule_seq


    def __compute_start(self, new_rule, prev_rule):
        # This function is to find where to start the computation
        # We use this mechanism to speed up the algorithm
        # before start, it is copy
        
        # At the first iteration prev_rule is empty
        if len(prev_rule) == 0:
            return 0
        start = 0
        match = False # Check if two non-empty rule list have different rule
        len_min = min(len(new_rule),len(prev_rule))
        
        for i in range(len_min):
            if prev_rule[i] != new_rule[i]:
                start = i
                match = True
                break    
        
        if match == False:
            # If no different rule, then the last rule must be removed/added
            start = max(len(new_rule),len(prev_rule))-1
        
        return start


    def __update_support(self, new_rule, curr_rule, curr_coverage):
        # This function generates new coverage using current rule to 
        # reduce computations
        
        # new rule, previous support, previous cover and a start point
        # start should not be larger than len(rule_list)-1 or len(support_map)
        
        new_coverage = Coverage()
        
        # Find start position by comparing new rule and previous rule
        start = self.__compute_start(new_rule, curr_rule)

        # Copy before start
        if start > 0:
            new_coverage.rule_cover.extend(curr_coverage.rule_cover[:start])
            new_coverage.rule_catch.extend(curr_coverage.rule_catch[:start])
            new_coverage.bbox_cover.extend(curr_coverage.bbox_cover[:start])
            new_coverage.A.extend(curr_coverage.A[:start])
            new_coverage.choose.extend(curr_coverage.choose[:start])
        
        # Compute after start
        for i in range(start, len(new_rule)): # Only do set computation from start

            if i == 0:
                new_coverage.rule_cover.append(self.cover_sets[new_rule[i]])
                new_coverage.rule_catch.append(self.cover_sets[new_rule[i]])
            else:
                new_coverage.rule_cover.append((self.cover_sets[new_rule[i]]) |  (new_coverage.rule_cover[i-1]))
                new_coverage.rule_catch.append((self.cover_sets[new_rule[i]]) & ~(new_coverage.rule_cover[i-1]))
            
            new_coverage.bbox_cover.append(~new_coverage.rule_cover[i])

            # Number of instance captured by mth list
            covered_num = sum(new_coverage.rule_catch[i])
            # Positive instance captured my mth list
            pos_label = self.Y&(new_coverage.rule_catch[i])

            # Rule m predicts positive class
            if sum(pos_label) > covered_num / 2:
                # The only changes in error are:
                #   bb predict neg error -> rule predict pos correct
                #   bb predict neg correct -> rule predict pos error
                delta_A = sum(self.Y&(~self.Yb)&new_coverage.rule_catch[i])
                delta_A -= sum((~self.Y)&(~self.Yb)&new_coverage.rule_catch[i])
                new_coverage.choose.append(1)
            # Rule m predicts negative class
            else:
                # The only changes in error are:
                #   bb predict pos error -> rule predict neg correct
                #   bb predict pos correct -> rule predict neg error
                delta_A = sum((~self.Y)&self.Yb&new_coverage.rule_catch[i])
                delta_A -= sum(self.Y&self.Y_c&new_coverage.rule_catch[i])
                new_coverage.choose.append(0)
            if i == 0:
                new_coverage.A.append(self.BLX_ACC + delta_A / self.N)
            else:
                new_coverage.A.append(new_coverage.A[i-1] + delta_A / self.N)

        return new_coverage
    

    def __obj_func(self, coverage):
        
        # Length of rulelist
        k = len(coverage.rule_cover)
        
        obj = (coverage.A[0] + self.BLX_ACC) * sum(coverage.rule_catch[0])#
        obj += sum([(coverage.A[i] + coverage.A[i-1]) * sum(coverage.rule_catch[i]) for i in range(1, k)])
        obj *= 0.5 / self.N
        obj -= self.alpha * k
        return obj



    def __simulated_annealing(self, init_rule_idx, iteration, T0):
        
        # Temperature
        temperature = T0
        
        # Optimal objective
        obj_best = 0
        
        # Initialization
        curr_coverage = Coverage()
        curr_rule_idx = init_rule_idx
        curr_coverage = self.__update_support(curr_rule_idx, [], curr_coverage)
        obj_curr = self.__obj_func(curr_coverage)

        # Main iteration
        if self.time_limit is not None:
            start = time.process_time()
        for t in tqdm(range(iteration), disable=not(self.print_progress)):
            # Randomly perturb the current rulelist
            new_rule_idx = self.__propose_rule(curr_rule_idx)

            # Update coverage of rule list
            new_coverage = self.__update_support(new_rule_idx, curr_rule_idx, curr_coverage)

            # Compute objective
            obj_new = self.__obj_func(new_coverage)
            
            if obj_new > obj_curr:
                # Accept the change
                obj_curr = obj_new
                curr_rule_idx = new_rule_idx
                curr_coverage = new_coverage
                # Update optimal solution
                if obj_new > obj_best:
                    obj_best = obj_new
                    rule_idx_best = new_rule_idx
                    coverage_best = new_coverage
            # Update worst solution with low probablity
            else:
                rand = random.random()
                accept_rate = math.exp((obj_new-obj_curr)/temperature)
                if accept_rate > rand:
                    # Accept the change
                    obj_curr = obj_new
                    curr_rule_idx = new_rule_idx
                    curr_coverage = new_coverage
            # Lower the temperature
            temperature = T0 / math.log(2 + t)

            if self.time_limit is not None:
                if time.process_time() - start > self.time_limit:
                    break
        return rule_idx_best, coverage_best, obj_best


    def fit(self, X, y, n_iteration=5000, init_temperature=0.001, random_state=42, 
                                            premined_rules=False, time_limit=None, print_progress=False):

        """
        Build a HyRS from the training set (X, y).

        Parameters
        ----------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary.

        y : np.array, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        n_iteration : number of iterations of the local search.

        Returns
        -------
        self : obj
        """
        # Set the seed for reproducability
        random.seed(random_state)
        self.time_limit = time_limit

        self.print_progress = print_progress
        
        # Store the training data
        self.df = X
        self.Y = y
        self.N = len(y)

        # If the feature are already rules that have been mined
        if premined_rules:
            self.all_rules = [[f] for f in X.columns]
            self.n_rules = X.shape[1]
            self.len_rules = [1 for _ in range(self.n_rules)]
        # Otherwise mine the rules
        # else:  #uncomment the whole block if you want to do rule mining (DONE BY ZIBA)
        #     _, prules, nrules = generate_rulespace(X, y, self.max_card, random_state=random_state)
        #     self.all_rules = prules + nrules
        #     self.n_rules = len(self.all_rules)
        #     self.len_rules = [len(self.all_rules[i]) for i in range(self.n_rules)]

        # Predictions and errors
        self.Y = bitarray(list(y))
        self.Yb = bitarray(list(self.bbox.predict(X)))
        self.Y_bberror = (self.Y^self.Yb)
        self.Y_c = ~self.Y_bberror  # to speed up
        self.BLX_ACC = sum(self.Y_c) / self.N # overall black box accuracy
        
        # Set on 
        init_rule_idx = random.sample(list(range(len(self.all_rules))), 3)

        # list of bitarrays indicating whether or not rule i covers example j
        cover_sets = [np.sum(self.df[self.all_rules[i]], axis=1)==self.len_rules[i] for i in range(self.n_rules)]
        self.cover_sets = [bitarray(list(cover_sets[i])) for i in range(self.n_rules)]

        T0 = init_temperature
        self.rule_idx, self.coverage, self.obj = self.__simulated_annealing(init_rule_idx, n_iteration, T0)

        return self


    # def predict_with_type(self, X):
    #     """
    #     ToDo
    #     Predict classifications of the input samples X, along with a boolean (one per example)
    #     indicating whether the example was classified by the interpretable part of the model or not.

    #     Arguments
    #     ---------
    #     X : pd.DataFrame, shape = [n_samples, n_features]
    #         The training input samples. All features must be binary, and the same
    #         as those of the data used to train the model.

    #     Returns
    #     -------
    #     ToDo
    #     """

    #     test_Yb = self.bbox.predict(X)
    #     # use test data to test the rule list
    #     output_rules = [self.premined_rules[self.rule[i]] for i in range(len(self.rule))]
    #     catch_list = np.array([-1]*X.shape[0])
    #     # to show observation is caught by which rule
    #     for i in range(X.shape[0]):
    #         for j in range(len(output_rules)):
    #             match = True
    #             for condition in output_rules[j]:
    #                 if X.iloc[i][condition] == 0:
    #                     match = False
    #                     break
    #             if match == True:
    #                 catch_list[i] = j
    #                 break
        
    #     test_cover_rate = [0]*len(output_rules)
    #     blx_cover_rate = [0]*len(output_rules)
    #     test_acc = []
    #     blx_acc = []
        
    #     rule_coverd_set = set()
    #     blx_cover_set = set(range(X.shape[0]))


    #     Yhat = []
    #     covered = []
        
    #     for i in range(len(output_rules)):
    #         # observation num caught by rule i
    #         rule_catch = np.where(catch_list==i)[0]
    #         # the accumulated rules catch by rule list
    #         rule_coverd_set = rule_coverd_set.union(set(rule_catch))
    #         # the left part is then caught by blx model
    #         blx_cover = blx_cover_set.difference(rule_coverd_set)
    #         # blx cover rate
    #         blx_cover_rate[i] = len(blx_cover)/(X.shape[0]+0.0001)
    #         # cover rate and accuracy of rules
    #         test_cover_rate[i] = len(rule_catch)/(X.shape[0] + 0.0001)
        
        
        
    def predict_with_type(self, X):
        """
        Predict labels for X and indicate whether prediction
        comes from interpretable rule list (1) or black-box (0).
        """

        N = X.shape[0]

        # Default: black-box
        y_pred = np.array(self.bbox.predict(X))
        pred_type = np.zeros(N, dtype=int)

        uncovered = np.ones(N, dtype=bool)

        # Apply rules sequentially (EXACTLY like test())
        for i, r_idx in enumerate(self.rule_idx):
            rule_features = self.all_rules[r_idx]

            # Explicit AND over literals
            rule_cover = np.ones(N, dtype=bool)
            for f in rule_features:
                rule_cover &= (X[f].values == 1)

            # Only catch uncovered points
            rule_catch = rule_cover & uncovered

            if np.any(rule_catch):
                y_pred[rule_catch] = self.coverage.choose[i]
                pred_type[rule_catch] = 1
                uncovered[rule_catch] = False

        return y_pred, pred_type



    def test(self, X, y):
        ybb = self.bbox.predict(X)
        N  = len(y)
        # Use test data to test the rule list
        output_rules = [self.all_rules[self.rule_idx[i]] for i in range(len(self.rule_idx))]
        catch_list = np.array([-1]*N)
        
        # Which observation is caught by which rule
        for i in range(N):
            for j in range(len(output_rules)):
                match = True
                for condition in output_rules[j]:
                    if X.iloc[i][condition] == 0:
                        match = False
                        break
                if match == True:
                    catch_list[i] = j
                    break
                    
        rule_coverage = [0]*len(output_rules)
        bb_coverage = [0]*len(output_rules)
        rule_correct = []
        bb_correct = []
        rule_cover_set = set()
        blx_cover_set = set(range(N))
        
        for i in range(len(output_rules)):
            # Observations caught by rule i
            rule_catch = np.where(catch_list==i)[0]
            # Cummulated instance caugth by rule list
            rule_cover_set = rule_cover_set.union(set(rule_catch))
            # The rest is send to the bb model
            bb_cover = blx_cover_set.difference(rule_cover_set)
            # BB coverage
            bb_coverage[i] = len(bb_cover) / N 
            # BB accuracy
            bb_correct.append(sum(ybb[list(bb_cover)]==y[list(bb_cover)]))
            # cover rate and accuracy of rules
            rule_coverage[i] = 1 - bb_coverage[i]#len(rule_catch)/(test_data.shape[0] + 0.0001)
            rule_correct.append(sum(y[rule_catch] == self.coverage.choose[i]))
        
        # The overall accuracy of hybrid models
        test_overall_acc = [(sum([rule_correct[i] for i in range(j+1)]) + bb_correct[j]) / N for j in range(len(output_rules))]
        return output_rules, rule_coverage, test_overall_acc



    def get_description(self, X, y):
        """
        Return a string description of the Hybrid Model. Datasets are provided 
        to compute the accuracy and coverage

        Arguments
        ---------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and 
            must be the same as those of the data used for training.
        y : np.array, shape = [n_samples]
            The target values for the training input. Must be binary.

        Returns
        -------
        s : str
            The string description.
        """
        output_rules, rule_coverage, acc = self.test(X, y)

        s = "CRL\n"
        s += " ACC ,  C  ,  Rule\n"
        for (i, j, k, p) in zip(acc, output_rules, rule_coverage, self.coverage.choose):
            s += f"{i:.3f} {k:.3f} {j}=>{p}\n"
        s += "\n"
        
        return s



def accumulate(iterable, func=operator.add):
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total