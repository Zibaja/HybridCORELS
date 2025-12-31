import json
from pathlib import Path
import pandas as pd
import numpy as np


Data_dir = Path('/Users/ziba/programming/optimization/HybridCorels-julien/HybridCORELS/examples/')

def read_json(datadir):
    with open(datadir, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def age_data_modification(X,features):
    """modify dataset to have mutually exclusive age groups"""
    df_x = pd.DataFrame(X, columns=features)
    df_x['Age=24-25'] = df_x.apply(lambda row: 1 if row['Age=18-25']==1 & row['Age=24-30']==1  else 0, axis=1)
    df_x['Age=18-23']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=18-25'], axis=1)
    df_x['Age=26-30']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=24-30'], axis=1)
    df_x['Age=26-29']= df_x.apply(lambda row: 0 if row['Age>=30']==1 & row['Age=26-30']==1 else row['Age=26-30'], axis=1)
    return np.array(df_x), df_x.columns.tolist()


class DemographicData():
    def __init__(self, X,y, features, condition):
        self.X = X
        self.features = features
        self.condition = condition
        self.y = y
    
    def filter_data(self):
        filtered_data = self.X[:, [i for i,j in enumerate(self.features) if j in self.condition]]
        if len (self.condition)==1:
            filtered_data = filtered_data.ravel()
        return filtered_data
    
    def get_condition_freq(self):
        condition_count = self.filter_data().sum(axis=0)
        freq = (condition_count/self.X.shape[0])*100
        return dict(zip(self.condition, freq))

    def corr_subgroups(self):
        correlation= {}
        for i,j in enumerate(self.condition):
            corr = np.corrcoef(self.y,self.filter_data()[:,i])[0,1]
            correlation[j]=corr
        return correlation




# faireness evaluation add by ziba

class FairnessMeasure():
    def __init__(self,X ,features: list, condition:list):
        """
        This class computes fairness measures based on transparency for a given condition.
        X: np.ndarray
            it could be train or test data.
        features: list
            List of feature names corresponding to the columns of X.
        condition: list
            List of feature names that define the condition. The condition is the intersection of these features being true (1)."""
        self.X = X
        self.features = features
        self.condition = condition
        self.cond_indices = self.set_condition()
    #slice the data based on the condition
    def set_condition(self):  
        indices_of_features = [i for i,j in enumerate(self.features) if j in self.condition]
        condition_indices = np.all((self.X[:,indices_of_features]),axis=1)
        return condition_indices
    
    #get frequency of a condition 
    def get_condition_freq (self):
       return float((self.set_condition().sum()/self.X.shape[0])*100)
    
    # compute of percentage of interpretable samples for the given condition
    def compute_fairness(self, preds_types,complement = False):
        condition_indices = self.cond_indices
        if complement:
            condition_indices = np.logical_not(condition_indices)
        total_count = condition_indices.sum() # total number of samples satisfying the condition
        interpretable_count = preds_types[condition_indices].sum() # number of samples satisfying the condition going through the interpretable part
        percentage_interpretable = (interpretable_count / total_count) * 100 if total_count > 0 else 0
        return {
            "condition": self.condition,
            "total_count": int(total_count),
            "interpretable_count": int(interpretable_count),
            "percentage_interpretable": float(percentage_interpretable)#'%.2f' % percentage_interpretable
        }
    

    def confusion_matrix(self, pred, y_true, pred_types, cond_indices, detailed = True):
        """This fucntions return the confusion matrix for each condition 
        and for both interpretable and black box part

        Args:
            pred (numpy array): output prediction
            y_true (numpy array): target label
            pred_types (numpy array): an array that shows if an instance is interpreted by rule list or by the BB
            cond_indices (numpy array): indices of instaces of a subgroup
            detailed (bool, optional): True if a detailed confusion matrix per interpretable and BB is wanted. Defaults to False.
        """
        if detailed:
            TP_interpret= np.sum((y_true[cond_indices]==pred[cond_indices])
                                        & (y_true[cond_indices]==1)& (pred_types[cond_indices]==1))
            TP_BB = np.sum((y_true[cond_indices]==pred[cond_indices])
                                        & (y_true[cond_indices]==1)& (pred_types[cond_indices]==0))
            TN_interpret = np.sum((y_true[cond_indices]==pred[cond_indices]) & (y_true[cond_indices]==0) 
                                        & (pred_types[cond_indices]==1))
            TN_BB = np.sum((y_true[cond_indices]==pred[cond_indices]) & (y_true[cond_indices]==0) 
                                        & (pred_types[cond_indices]==0))
            FP_interpret = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==1)
                                        & (pred_types[cond_indices]==1))
            FP_BB = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==1)
                                        & (pred_types[cond_indices]==0))
            FN_interpret = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==0)
                                        & (pred_types[cond_indices]==1))
            FN_BB = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==0)
                                        & (pred_types[cond_indices]==0))
            return {'Interpretable':np.array([[TN_interpret, FP_interpret],
                        [FN_interpret, TP_interpret]]), 'Blackbox': np.array([[TN_BB,FP_BB],
                        [FN_BB, TP_BB]])}
        else:
            TP = np.sum((y_true[cond_indices]==pred[cond_indices]) & (y_true[cond_indices]==1))
            TN = np.sum((y_true[cond_indices]==pred[cond_indices]) & (y_true[cond_indices]==0))
            FP = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==1))
            FN = np.sum((y_true[cond_indices]!=pred[cond_indices]) & (pred[cond_indices]==0))

            print("Confusion Matrix")
            print("----------------")
            print(f"           \tPredicted Negative    Predicted positive")
            print(f"Actual Negative\t\t{TN:4d}\t      {FP:4d}")
            print(f"Actual Positive\t\t{FN:4d}\t      {TP:4d}")
            return np.array([[TN, FP],
                        [FN, TP]])


#This function get the initial json data file and extract the TPR for each seed and each subgroup

def get_TPR (data):
    """This function calulates the TPR ratio from the confusion matrix

    Args:
        data (dict): a portion of data that have all conditions as key

    Returns:
        dict: a dictionary with all conditions as key 
    """
    TPR_info = {cond:{'TPR_overal':[], 'TPR_T':[], 'TPR_B':[]} for cond in data.keys()}
    for cond in data.keys(): #this is for all conditions in data
        TPR_overal=[]
        TPR_T=[]
        TPR_BB=[]
        first_metric = next (iter(data[cond]))
        n_seeds = len(data[cond][first_metric])
        for i in range(n_seeds):
            TP_overall = (data[cond]['TP']['T'][i]+ data[cond]['TP']['B'][i])
            FN_overall = (data[cond]['FN']['T'][i]+ data[cond]['FN']['B'][i])
            if (TP_overall+FN_overall) !=0:
                TPR = TP_overall/(TP_overall+FN_overall)
            else:
                TPR= 0
            TPR_overal.append(TPR)
            TP_T = data[cond]['TP']['T'][i]
            FN_T = data[cond]['FN']['T'][i]
            if (TP_T+ FN_T) !=0:
                TPR_Trans = TP_T/ (TP_T+ FN_T )
            else:
                TPR_Trans = 0
            TPR_T.append(TPR_Trans)
            TP_B = data[cond]['TP']['B'][i]
            FN_B = data[cond]['FN']['B'][i]
            if (TP_B+FN_B) != 0:
                TPR_Blackbox = TP_B / (TP_B+FN_B)
            else:
                TPR_Blackbox = 0
            TPR_BB.append(TPR_Blackbox)

        TPR_info[cond]['TPR_overal']= TPR_overal
        TPR_info[cond]['TPR_T']= TPR_T
        TPR_info[cond]['TPR_B']= TPR_BB
    return TPR_info


def compute_global_fairness (TPR_info, condition,model_part= None): # 
    """this function computes the global fairness measure which is the maximum gap 
    between TPR of each demographic group for each seed. Also we can compute the overall fairness for the Hybrid model 
    or only for the interpretable part or only for the BlackBox 

    Args:
        TPR_info (dict): This dictionary include the TPR for all conditions and all seeds
        condition (list): this is the dmographic subgroup for which we measure the fairness
        model_part (str, optional): None if overall performance is wanted, 'B' for BB fairness and 'T' for 
        interpretable part fairness . Defaults to None.

    Returns:
        numpy array: array of global fairness measure for all seeds
    """
    fairness_gaps = []
    if model_part == None:
        part = 'TPR_overal'
    elif model_part == 'T':
        part = 'TPR_T'
    elif model_part == 'B':
        part = 'TPR_B'
    first_group = next(iter(TPR_info))
    # pick the first metric key
    first_metric = next(iter(TPR_info[first_group]))
    n_seeds = len(TPR_info[first_group][first_metric])
    for i in range (n_seeds):
        gap = max([abs(TPR_info[g1][part][i] - TPR_info[g2][part][i])\
        for g1 in condition for g2 in condition if g1!= g2 ])
        fairness_gaps.append(gap)

    return np.array(fairness_gaps)
        
def compute_signed_global_fairness (TPR_info, condition,model_part): # 
    """this function computes the global fairness measure which is the maximum gap 
    between TPR of each demographic group for each seed. Also we can compute the overall fairness for the Hybrid model 
    or only for the interpretable part or only for the BlackBox 

    Args:
        TPR_info (dict): This dictionary include the TPR for all conditions and all seeds
        condition (list): this is the dmographic subgroup for which we measure the fairness
        model_part (str, optional): None if overall performance is wanted, 'B' for BB fairness and 'T' for 
        interpretable part fairness . Defaults to None.

    Returns:
        list: list of global fairness measure for all seeds
    """
    if len(condition)!=2:
        raise ValueError('Signed TPR difference is over two subgroups')
    g1 = condition[0]
    g2 = condition[1]
    fairness_gaps = []
    if model_part == None:
        part = 'TPR_overal'
    elif model_part == 'T':
        part = 'TPR_T'
    elif model_part == 'B':
        part = 'TPR_B'

    fairness_gaps = np.array(TPR_info[g1][part])-np.array(TPR_info[g2][part])
    return np.array(fairness_gaps)



def statistics (array):
    mean = np.mean(array)
    std = np.std(array)
    se = np.std(array) / np.sqrt(len(array))
    return mean, std, se

def paired_subgroups (subgroups):
    return [[x, y] for i, x in enumerate(subgroups) for y in subgroups[i+1:]]