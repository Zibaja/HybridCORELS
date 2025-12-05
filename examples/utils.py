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


def get_condition_freq (X, features, condition):
    """
    this function calculates the frequency rate of each condition; if conditions are of one cardinalty 
    and given as a list of conditions 
    example : ['Gender=Male', 'neg_Gender=Male']
    """
    filtered_data = X[:, [i for i,j in enumerate(features) if j in condition]]
    condition_count = filtered_data.sum(axis=0)
    freq = (condition_count/X.shape[0])*100
    return dict(zip(condition, freq))



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
        condition_indices = self.set_condition()
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
    
