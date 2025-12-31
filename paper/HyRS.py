import pandas as pd 
import numpy as np
from numpy.random import random
from bisect import bisect_left
from random import sample, seed
from sklearn.metrics import confusion_matrix
import operator
from dataclasses import dataclass
from tqdm import tqdm
import time

#from rule_mining import generate_rulespace, screen_rules #uncomment for rule mining (DONE BY ZIBA)

@dataclass
class Coverage:
    """ Store results on the Coverage of a RuleSet """
    p: np.array # Is instance i covered by R+ ?
    n: np.array # Is instance i covered by R- ?
    overlap: np.array   # Is instance i covered by both R+ and R- ?
    pcovered: np.array  # Is instance i ONLY covered by R+ ?
    ncovered: np.array  # Is instance i ONLY covered by R- ?
    covered: np.array   # Which instance are covered by R+ U R- ?


class HybridRuleSetClassifier(object):
    def __init__(self, black_box_classifier, min_support=5, max_card=2, n_rules=5000, alpha=1, beta=0.1):
        """Hybrid Rule Set/Black-box based classifier.

        This class implements an Hybrid interpretable/black-box model.
        It considers that the black-box model has already been fitted and then fits a ruleset on top of
        it in order to gain free transparency. At inference, instances covered by the rules are labeled
        according to those rules while instances not covered are send to the black box for prediction.

        Attributes
        ----------

        black_box_classifier: already fitted classifier

        min_support: minimal support of the rules (in %)

        max_card: maximal cardinality of the rules

        n_rules: number of rules in the rule-space to consider

        alpha: coefficient to weight the Interpretability term in the objective

        beta: coefficient to weight the Coverage term in the objective

        References
        ----------
        Wang, T. (2019, May). Gaining free or low-cost interpretability with interpretable 
        partial substitute. In International Conference on Machine Learning 
        (pp. 6505-6514). PMLR.
        """

        self.min_support = min_support
        self.max_card = max_card
        self.n_rules = n_rules
        self.black_box_classifier = black_box_classifier
        self.alpha = alpha
        self.beta = beta
        


    def fit(self, X, y, n_iteration=5000, T0=0.01, interpretability='size', print_progress=False, 
                                                random_state=42, premined_rules=False, time_limit=None):
        """
        Build a HyRS from the training set (X, y).

        Parameters
        ----------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary.

        y : np.array, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        n_iteration : number of iterations of the local search.

        premined_rules : Boolean
            Whether or not the features of X are already premined rules

        Returns
        -------
        self : obj
        """
        # Set the seed for reproducability
        np.random.seed(random_state)
        seed(random_state)

        # Store the training data
        self.df = X
        self.Y = y
        self.N = len(y)
        self.Yb = self.black_box_classifier.predict(X)

        self.premined_rules = premined_rules
        # If the feature are already rules that have been mined
        if premined_rules:
            all_rules = list(X.columns)
            self.prules = all_rules
            self.pRMatrix = X.to_numpy()
            self.nrules = all_rules
            self.nRMatrix = self.pRMatrix
        # Otherwise mine the rules
        # else:  #uncomment if you want to premine rules (DONE BY ZIBA)
        #     _, prules, nrules = generate_rulespace(X, y, self.max_card, random_state=random_state)
        #     self.prules, self.pRMatrix = screen_rules(prules, X, self.n_rules // 2, self.min_support)
        #     self.nrules, self.nRMatrix = screen_rules(nrules, X, self.n_rules // 2, self.min_support)

        # Setup
        self.maps = []
        int_flag = int(interpretability =='size')
        obj_curr = 1000000000
        obj_opt = obj_curr

        # We store the rule sets using their indices
        prs_curr = sample(list( range(len(self.prules)) ), 3)
        nrs_curr = sample(list( range(len(self.nrules)) ), 3)
        self.maps.append([-1, obj_curr, prs_curr, nrs_curr]) #[iter, obj, prs, nrs])

        # Coverage of P-N rules
        coverage_curr = self.__compute_rules_coverage(prs_curr, nrs_curr)

        # Compute the loss function
        Yhat_curr, TP, FP, TN, FN  = self.__compute_loss(coverage_curr)

        # Count the number of features
        nfeatures = len(np.unique([con.split('_')[0] for i in prs_curr for con in self.prules[i]])) + \
                    len(np.unique([con.split('_')[0] for i in nrs_curr for con in self.nrules[i]]))

        # Compute the objective function
        # New objective function
        o1_curr, o2_curr, o3_curr = self.__compute_objective(FN, FP, int_flag, nfeatures, \
                                                            prs_curr, nrs_curr, coverage_curr)
        obj_curr = o1_curr + self.alpha * o2_curr + self.beta * o3_curr
        

        # Main Loop
        self.actions = []
        if time_limit is not None:
            start = time.process_time()
        for iter in tqdm(range(n_iteration), disable=not(print_progress)):
            
            # Propose new RuleSets
            prs_new, nrs_new, coverage_new = self.__propose_rs(Yhat_curr, prs_curr, nrs_curr, coverage_curr)
    
            # Compute the new loss
            Yhat_new, TP, FP, TN, FN = self.__compute_loss(coverage_new)
            
            # New number of features
            nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + \
                        len(np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
            
            # New objective function
            o1_new, o2_new, o3_new = self.__compute_objective(FN, FP, int_flag, nfeatures, \
                                                            prs_new, nrs_new, coverage_new)
            obj_new = o1_new + self.alpha * o2_new + self.beta * o3_new

            # Decrease Temperature
            T = T0 ** (iter / n_iteration)
            # Acceptance Probability
            alpha = np.exp(float(obj_curr - obj_new) / T)

            # We decreased the optimal objective
            if obj_new < obj_opt:
                prs_opt = prs_new
                nrs_opt = nrs_new
                obj_opt = obj_new
            
            # Accept the change is probability alpha
            if random() <= alpha:
                # Update current solution
                prs_curr = prs_new
                nrs_curr = nrs_new
                obj_curr = obj_new
                coverage_curr = coverage_new
                Yhat_curr = Yhat_new
                self.maps.append([iter, obj_new, prs_new, nrs_new])

                if print_progress:
                    print(f"\n** iter={iter} ** \n {obj_new:.3f}(obj) = " +\
                        f"{o1_new:.3f}(error) + alpha * {o2_new:d}(interp) + beta * {o3_new:.3f}(1-coverage)")
                    
            if time_limit is not None:
                if time.process_time() - start > time_limit:
                    break
        
        # Save the optimal rule sets as attributes
        self.positive_rule_set = prs_opt
        self.negative_rule_set = nrs_opt

        return self



    def __compute_rules_coverage(self, prs, nrs):
        p = np.sum(self.pRMatrix[:, prs], axis=1) > 0  # Is instance i covered by R+ ?
        n = np.sum(self.nRMatrix[:, nrs], axis=1) > 0  # Is instance i covered by R- ?
        overlap = np.multiply(p, n)    # Is instance i covered by both R+ and R- ?
        pcovered = p ^ overlap         # Is instance i ONLY covered by R+ ?
        ncovered = n ^ overlap         # Is instance i ONLY covered by R- ?
        covered  = np.logical_or(p, n) # Which instance are covered by R+ U R- ?
        assert sum(overlap) <= sum(covered)
        return Coverage(p, n, overlap, pcovered, ncovered, covered)



    def __compute_loss(self, coverage):
        # p: is x covered by R+ ?
        # covered : is x covered by R+ U R- ?
        Yhat = np.zeros(int(self.N))
        Yhat[coverage.p] = 1
        Yhat[~coverage.covered] = self.Yb[~coverage.covered]
        TN, FP, FN, TP = confusion_matrix(Yhat, self.Y).ravel()
        return  Yhat, TP, FP, TN, FN



    def __compute_objective(self, FN, FP, int_flag, nfeatures, prs, nrs, coverage):
        return (FN + FP) / self.N, \
                (int_flag *(len(prs) + len(nrs)) + (1 - int_flag) * nfeatures),\
                sum(~coverage.covered) / self.N



    def __propose_rs(self, Yhat, prs, nrs, coverage):# vt, print_message = False):
        # Error because of the interpretable model
        incorr = np.where(Yhat[coverage.covered] != self.Y[coverage.covered])[0]
        # Error because of the blackbox model
        incorrb = np.where(Yhat[~coverage.covered] != self.Y[~coverage.covered])[0]
        ex = -1
        # Covering all examples
        if sum(coverage.covered) == self.N:
            move = ['cut']
            self.actions.append(0)
            if len(prs) == 0:
                sign = [0]
            elif len(nrs) == 0:
                sign = [1]
            else:
                sign = [int(random()<0.5)]
        # Covering no examples
        if sum(coverage.covered) == 0:
            move = ['extend']
            self.actions.append(1)
            sign = [int(random()<0.5)]
        # # What the heck ???
        # elif len(incorr) == 0 and (len(incorrb)==0 or sum(coverage.overlap)==self.N): #\or sum(coverage.overlap) > sum(coverage.covered):
        #     self.actions.append(1)
        #     move = ['cut']
        #     sign = [int(random()<0.5)]
        # elif (len(incorr) == 0 and (sum(covered)>0)) or len(incorr)/sum(covered) >= len(incorrb)/sum(~covered):
        #     if print_message:
        #         print(' ===== 2 ===== ')
        #     self.actions.append(2)
        #     ex = sample(list(np.where(~covered)[0]) + list(np.where(overlapped)[0]),1)[0] 
        #     if overlapped[ex] or len(prs) + len(nrs) >= (vt + self.beta)/self.alpha:
        #         # print('2')
        #         move = ['cut']
        #         sign = [int(random()<0.5)]
        #     else:
        #         # print('3')
        #         move = ['expand']
        #         sign = [int(random()<0.5)]
        else:
            t = random()
            if t < 1./3: # Try to decrease errors
                self.actions.append(3)
                ex = sample(list(incorr) + list(incorrb), 1)[0]
                if ex in incorr: # Incorrectly classified by the interpretable model
                    if self.Y[ex] == 0: # Negative example
                        move = ['cut']  # Remove the positive rule that covers it
                        sign = [1]
                    else: # Positive example
                        if random() < 0.5: # Add a rule to cover it
                            move = ['add']
                            sign = [1]
                        else: # Remove a negative rule that covers it
                            move = ['cut']
                            sign = [0]
                    #rs_indicator = (coverage.pcovered[ex]).astype(int) # covered by prules
                    #if random() < 0.5:
                    #    move = ['cut']
                    #    sign = [rs_indicator]
                    #else:
                    #    move = ['cut', 'add']
                    #    sign = [rs_indicator, rs_indicator]
                # elif overlapped[ex]: 
                #     if random()<0.5 :
                #         # print('5')
                #         move = ['cut']
                #         sign = [1 - self.Y[ex]]
                #     else:
                #         # print('6')
                #         move = ['cut','add']
                #         sign = [1 - self.Y[ex],1 - self.Y[ex]]
                else: # Incorrectly classified by the black box model
                    move = ['add']
                    sign = [int(self.Y[ex]==1)] # Add a consistent rule that covers it
            elif t < 2./3: # decrease coverage
                self.actions.append(4)
                move = ['cut']
                sign = [round(random())]
            else: # increase coverage
                self.actions.append(5)
                move = ['expand']
                sign = [round(random())]
                # if random()<0.5:
                #     move.append('add')
                #     sign.append(1-rs_indicator)
                # else:
                #     move.extend(['cut','add'])
                #     sign.extend([1-rs_indicator,1-rs_indicator])

        # Update the sets R+ and R-
        #for j in range(len(move)):
        j = 0
        if sign[j] == 1:
            prs_new = self.__action(move[j], 1, ex, prs, Yhat, coverage.pcovered)
            nrs_new = nrs
        else:
            prs_new = prs
            nrs_new = self.__action(move[j], 0, ex, nrs, Yhat, coverage.ncovered)

        coverage_new = self.__compute_rules_coverage(prs_new, nrs_new)

        return prs_new, nrs_new, coverage_new


    def __action(self, move, rs_indicator, ex, rules, Yhat, covered):
        RMatrix = self.pRMatrix if rs_indicator else self.nRMatrix
        Y = self.Y if rs_indicator else 1 - self.Y

        # Removing a rule
        if move == 'cut' and len(rules) > 0:
            # Remove a rule to decrease the error at ex
            if random() < 0.25 and ex >= 0:
                # Remove a rule that covers ex
                candidate = list(set(np.where(RMatrix[ex, :]==1)[0]).intersection(rules))
                if len(candidate) == 0:
                    candidate = rules
                cut_rule = sample(candidate, 1)[0]
            # Remove any rule
            else:
                p = []
                all_sum = np.sum(RMatrix[:, rules], axis=1)
                # Compute the precision of the RuleSet if we remove a rule
                for index, rule in enumerate(rules):
                    Yhat = ((all_sum - np.array(RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = confusion_matrix(Yhat, Y).ravel()
                    if TP+FP == 0:
                        p.append(0)
                    else:
                        p.append(TP.astype(float) / (TP + FP))
                p = np.exp(np.array([x - min(p) for x in p]))
                p = np.insert(p, 0, 0)
                p = np.array(list(accumulate(p)))
                # Sample rules with Softmax prob based on precision
                if p[-1] == 0:
                    cut_rule = sample(rules, 1)[0]
                else:
                    p = p / p[-1]
                    index = find_lt(p, random())
                    cut_rule = rules[index]
            # Remove the selected rule
            return_rules = [rule for rule in rules if not rule == cut_rule]

        # We add a rule to decrease the error
        elif move == 'add' and ex >= 0:
            score_max = -self.N * 10000000
            # Add a rule with sign consistent with the target to cover the instance
            if self.Y[ex] * rs_indicator + (1 - self.Y[ex]) * (1 - rs_indicator) == 1:
                # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
                select = list(np.where(RMatrix[ex])[0])
            else:
                # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
                select = list(np.where(~RMatrix[ex])[0])
            self.select = select
            if len(select) > 0:
                if random() < 0.25:
                    add_rule = sample(select, 1)[0]
                else: 
                    # cover = np.sum(RMatrix[(~covered)&(~covered2), select],axis = 0)
                    # =============== Use precision as a criteria ===============
                    # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                    # mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
                    # TP = np.sum(mat,axis = 1)
                    # FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                    # TN = np.sum(Y[Yhat_neg_index]==0)-FP
                    # FN = sum(Y[Yhat_neg_index]) - TP
                    # p = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select]
                    # add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                    # =============== Use objective function as a criteria ===============
                    # What is going on here ???
                    for ind in select:
                        z = np.logical_or(RMatrix[:, ind], Yhat)
                        TN, FP, FN, TP = confusion_matrix(z, self.Y).ravel()
                        score = FP + FN - self.beta * sum(RMatrix[~covered ,ind])
                        if score > score_max:
                            score_max = score
                            add_rule = ind
            return_rules = rules + [add_rule]
        
        # Expand by adding any rule
        else:
            candidates = [x for x in range(RMatrix.shape[1])]
            select = list(set(candidates).difference(rules))
            if random() < 0.25:
                add_rule = sample(select, 1)[0]
            else:
                # What is going on in here !!!????
                # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                Yhat_neg_index = np.where(~covered)[0]
                mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1, 1),select].transpose(), Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat, axis=1)
                FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1, 1), select], axis=0) - TP)
                TN = np.sum(Y[Yhat_neg_index]==0) - FP
                FN = sum(Y[Yhat_neg_index]) - TP
                score = (FP + FN) + self.beta * (TN + FN)
                # score = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select] # using precision as the criteria
                add_rule = select[sample(list(np.where(score==min(score))[0]), 1)[0]] 
            
            return_rules = rules + [add_rule]
        return return_rules



    def predict_with_type(self, X):
        """
        Predict classifications of the input samples X, along with a boolean (one per example)
        indicating whether the example was classified by the interpretable part of the model or not.

        Arguments
        ---------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the same
            as those of the data used to train the model.

        Returns
        -------
        p, t : array of shape = [n_samples], array of shape = [n_samples].
            p: The classifications of the input samples
            t: The part of the Hybrid model which decided for the classification (1: interpretable part, 0: black-box part).
        """
        # Black box model
        Yb = self.black_box_classifier.predict(X)

        prules = [self.prules[i] for i in self.positive_rule_set]
        nrules = [self.nrules[i] for i in self.negative_rule_set]

        if not self.premined_rules:
            # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
            Xn = 1 - X
            Xn.columns = ['neg_' + name.strip() for name in X.columns]
            X_test = pd.concat([X, Xn], axis=1)
        else:
            X_test = X

        # Interpretable model
        if len(prules):
            # Does R+ cover these instances
            p = [[] for _ in prules]
            for i, rule in enumerate(prules):
                if not self.premined_rules:
                    p[i] = (np.sum(X_test[list(rule)], axis=1)==len(rule)).astype(int)
                else:
                    p[i] = X_test[rule]
            p = (np.sum(p, axis=0) > 0).astype(int)
        else:
            p = np.zeros(len(Yb))
        if len(nrules):
            # Does R- cover these instances
            n = [[] for _ in nrules]
            for i, rule in enumerate(nrules):
                if not self.premined_rules:
                    n[i] = (np.sum(X_test[list(rule)], axis=1)==len(rule)).astype(int)
                else:
                    n[i] = X_test[rule]
            n = (np.sum(n, axis=0) > 0).astype(int)
        else:
            n = np.zeros(len(Yb))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = np.logical_or(p, n)

        # Black box model
        Yhat = Yb
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat, covered



    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and 
            must be the same as those of the data used for training.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        return self.predict_with_type(X)[0]
    


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

        s = "HyRS\n"
        positive_rules = [self.prules[i] for i in self.positive_rule_set]
        negative_rules = [self.nrules[i] for i in self.negative_rule_set]
        s += f"Positive rules : \n {positive_rules}\n"
        s += f"Negative rules : \n {negative_rules}\n"

        # Test performance
        yhat, covered_index = self.predict_with_type(X)
        s += f"HyRS Accuracy : {np.mean(yhat == y):.3f}\n"
        s += f"Coverage of RuleSet : {np.sum(covered_index) / len(covered_index):.3f}\n"
        
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


def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0