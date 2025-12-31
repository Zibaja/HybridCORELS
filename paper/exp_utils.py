import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#from rule_mining import mine_rules_preprocessing

import os 
import numpy as np 

random_state_value = 42
test_size_ratio = 0.3

# # Load the new Adult dataset (using the provided folktables module)
# from folktables import ACSDataSource, ACSEmployment, employment_filter


# categories = {
#     "AGEP" : {
#         1.0 : "age_low",
#         2.0 : "age_medium",
#         3.0 : "age_high"
#     },
#     "SCHL" : {
#         1.0: "No schooling completed",
#         2.0: "Nursery school, preschool",
#         3.0: "Kindergarten",
#         4.0: "Grade 1",
#         5.0: "Grade 2",
#         6.0: "Grade 3",
#         7.0: "Grade 4",
#         8.0: "Grade 5",
#         9.0: "Grade 6",
#         10.0: "Grade 7",
#         11.0: "Grade 8",
#         12.0: "Grade 9",
#         13.0: "Grade 10",
#         14.0: "Grade 11",
#         15.0: "12th grade - no diploma",
#         16.0: "Regular high school diploma",
#         17.0: "GED or alternative credential",
#         18.0: "Some college, but less than 1 year",
#         19.0: "1 or more years of college credit, no degree",
#         20.0: "Associate's degree",
#         21.0: "Bachelor's degree",
#         22.0: "Master's degree",
#         23.0: "Professional degree beyond a bachelor's degree",
#         24.0: "Doctorate degree",
#     },
#     "MAR": {
#         1.0: "Married",
#         2.0: "Widowed",
#         3.0: "Divorced",
#         4.0: "Separated",
#         5.0: "Never married or under 15 years old",
#     },
#     "SEX": {1.0: "Male", 2.0: "Female"},
#     "RAC1P": {
#         1.0: "White alone",
#         2.0: "Black or African American alone",
#         3.0: "American Indian alone",
#         4.0: "Alaska Native alone",
#         5.0: (
#             "American Indian and Alaska Native tribes specified;"
#             "or American Indian or Alaska Native,"
#             "not specified and no other"
#         ),
#         6.0: "Asian alone",
#         7.0: "Native Hawaiian and Other Pacific Islander alone",
#         8.0: "Some Other Race alone",
#         9.0: "Two or More Races",
#     },
#     "ESP" : {
#         0.0 : "N/A (not own child of householder, and not child in subfamily)",
#         1.0 : "Living with two parent : both employed",
#         2.0 : "Living with two parent : Father employed",
#         3.0 : "Living with two parent : Mother employed",
#         4.0 : "Living with two parent : None employed",
#         5.0 : "Living with Father : Employed",
#         6.0 : "Living with Father : Not employed",
#         7.0 : "Living with Mother : Employed",
#         8.0 : "Living with Mother : Not employed",
#     },
#     "DIS" : {
#         1.0 : "Disability",
#         2.0 : "No disability"
#     },
#     "NATIVITY" : {
#         1.0 : "Native",
#         2.0 : "Foreign born"
#     },
#     "DREM" : {
#         1.0 : "Cognitive difficulty",
#         2.0 : "No Cognitive difficulty"
#     },
#     "RELP" : {
#         0.0 : "Reference person",
#         1.0 : "Husband/wife",
#         2.0 : "Biological son or daughter",
#         3.0 : "Adopted son or daughter",
#         4.0 : "Stepson or stepdaughter",
#         5.0 : "Brother or sister",
#         6.0 : "Father or mother",
#         7.0 : "Grandchild",
#         8.0 : "Parent-in-law",
#         9.0 : "Son-in-law or daughter-in-law",
#         10.0 : "Other relative",
#         11.0 : "Roomer or boarder",
#         12.0 : "Housemate or roommate",
#         13.0 : "Unmarried partner",
#         14.0 : "Foster child",
#         15.0 : "Other nonrelative",
#         16.0 : "Institutionalized group quarters population",
#         17.0 : "Noninstitutionalized group quarters population"
#     }
# }


# def binarize_age(X):
#     quantiles = np.quantile(X[:, 0], [0, 0.33, 0.66, 1])
#     for q in range(3):
#         index = np.where((quantiles[q] <= X[:, 0]) & (X[:, 0] <= quantiles[q+1]))[0]
#         X[index, 0] = q+1


# def generate_acs_data():
#     data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
#     acs_data = data_source.get_data(states=["TX"], download=True)
#     acs_data = employment_filter(acs_data)
#     features = ACSEmployment.features
#     X, y, _ = ACSEmployment.df_to_numpy(acs_data)
#     binarize_age(X)

#     # Reorder features according to categories dict
#     keep_features = list(categories.keys())
#     keep_features_idx = [ features.index(f) for f in keep_features]
#     X = X[:, keep_features_idx]

#     # OHE the categorical features
#     ohe = OneHotEncoder(sparse=False).fit_transform(X)
#     ohe_features = []
#     for feature_cat in categories.values():
#         ohe_features += list(feature_cat.values())

#     # Save the dataset
#     ohe_df = pd.DataFrame(np.column_stack((ohe, y)).astype(int), 
#                          columns=ohe_features+["Employed"])
#     filename = os.path.join("data", "acs_employ.csv")
#     ohe_df.to_csv(filename, encoding='utf-8', index=False)


def age_data_modification(X,features):
    """modify dataset to have mutually exclusive age groups"""
    df_x = pd.DataFrame(X, columns=features)
    df_x['Age=24-25'] = df_x.apply(lambda row: 1 if row['Age=18-25']==1 & row['Age=24-30']==1  else 0, axis=1)
    df_x['Age=18-23']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=18-25'], axis=1)
    df_x['Age=26-30']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=24-30'], axis=1)
    df_x['Age=26-29']= df_x.apply(lambda row: 0 if row['Age>=30']==1 & row['Age=26-30']==1 else row['Age=26-30'], axis=1)
    return np.array(df_x), df_x.columns.tolist()


def get_data(dataset, splits, max_card=2, min_support=1, n_rules=300, random_state_param=42):
    # # Generate the acs_data if it is not already there
    # if dataset == "acs_employ":# and not os.path.exists(f"data/{dataset}.csv"):
    #     generate_acs_data()

    # Mine the dataset set if it has not already been done
    if not os.path.exists(f"data/{dataset}_mined.csv"):

        df = pd.read_csv(f"data/{dataset}.csv", sep = ',')
        X = df.iloc[:, :-1]
        prediction = df.iloc[:, -1].name
        y = np.array(df.iloc[:, -1])

        # Mine the rules
        #X = mine_rules_preprocessing(X, y, max_card, min_support, n_rules) #uncomment when you need rulemining (DONE by ZIba)
        features = list(X.columns)
        X = np.array(X)

        # Save the dataset
        df = pd.DataFrame(np.column_stack((X, y)), columns=features+[prediction])
        df.to_csv(f"data/{dataset}_mined.csv", encoding='utf-8', index=False)

    # Rules have already been mined
    else:
        df = pd.read_csv(f"data/{dataset}_mined.csv", sep = ',')
        X = df.iloc[:, :-1]
        features = list(X.columns)
        X = np.array(X)
        prediction = df.iloc[:, -1].name
        y = np.array(df.iloc[:, -1])


    # Generate splits
    assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
    split_names = list(splits.keys())
    split_ratios = list(splits.values())
    assert np.sum(split_ratios) == 1, "The split ratios must sum up to one"
    X_dict = {}
    y_dict = {}
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=split_ratios[0],
                                          shuffle=True, random_state=random_state_param)
    X_dict[split_names[0]] = X_1
    y_dict[split_names[0]] = y_1
    if len(splits) == 2:
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
    else:
        sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=sub_ratio,
                                          shuffle=True, random_state=random_state_param)
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
        X_dict[split_names[2]] = X_3
        y_dict[split_names[2]] = y_3
    return X_dict, y_dict, features, prediction



def get_data_norulemining(dataset, splits, max_card=2, min_support=1, n_rules=300, random_state_param=42):
    # # Generate the acs_data if it is not already there
    # if dataset == "acs_employ":# and not os.path.exists(f"data/{dataset}.csv"):
    #     generate_acs_data()


    df = pd.read_csv(f"data/{dataset}_mined.csv", sep = ',')
    X = df.iloc[:, :-1]
    features = list(X.columns)
    X = np.array(X)
    prediction = df.iloc[:, -1].name
    y = np.array(df.iloc[:, -1])

    #compasdatasets need to be modified to have all age sections (DONE BY ZIBA)
    if dataset == 'compas':
        X, features = age_data_modification(X,features)


    # Generate splits
    assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
    split_names = list(splits.keys())
    split_ratios = list(splits.values())
    assert np.sum(split_ratios) == 1, "The split ratios must sum up to one"
    X_dict = {}
    y_dict = {}
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=split_ratios[0],
                                          shuffle=True, random_state=random_state_param)
    X_dict[split_names[0]] = X_1
    y_dict[split_names[0]] = y_1
    if len(splits) == 2:
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
    else:
        sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=sub_ratio,
                                          shuffle=True, random_state=random_state_param)
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
        X_dict[split_names[2]] = X_3
        y_dict[split_names[2]] = y_3
    return X_dict, y_dict, features, prediction

def to_df(X, features):
    df_X = {}
    for key, val in X.items():
        df_X[key] = pd.DataFrame(val, columns=features)
    return df_X

def computeAccuracyUpperBound(X, y, verbose=0):
    import pandas as pd
    import numpy as np
    """
    Parameters
    ----------
    X : Features vector
    y : Labels vector
    verbose : Int
    0 -> No display
    1 -> Minimal Display
    1 -> Debug (also performs additional checks)
    Returns
    -------
    Int, Array, Array
    Int : Minimum number of instances that can not be classified correctly due to dataset inconsistency
    Array of e_r: for each inconsistent group of examples r, e_r is a representative example of this group (its index in X)
    Array of k_r: k_r is the minimum number of instances that can not be classified correctly due to dataset inconsistency, among group r
    Array of i_r: all instances that will be misclassified in the best case (for all inconsistent group, those representing minority for their label)
    """
    representatives = []
    cardinalities = []
    misclassified = []
    values, counts = np.unique(X, axis=0, return_counts=True)
    values = values[counts > 1]
    counts = counts[counts > 1]
    if verbose >= 1:
        print("Found ", values.shape[0], " unique duplicates.")
    incorrCnt = 0
    for ii, anEl in enumerate(list(values)):
        occurences = np.where((X == anEl).all(axis=1))
        representant = occurences[0][0]
        if verbose >= 2:
            print("Value ", anEl, " appears ", counts[ii], " times. (CHECK = ", occurences[0].shape[0], ")")
            print("Occurences: ", occurences, "(representant is instance#", representant, ")")
            # Additional check
            if counts[ii] != occurences[0].shape[0]:
                exit(-1)
        labels = y[occurences[0]]
        if verbose >= 2:
            print(labels)
            # Additional check
            els = X[occurences[0]]
            elsC = np.unique(els, axis=0, return_counts=True)
            if elsC[0].shape[0] > 1:
                exit(-1)
        labelsData = np.unique(labels, return_counts = True)
        if labelsData[0].size > 1:
            if labelsData[0].size != 2: # only two possible values as we work with binary labels -> this case should never happen
                exit(-1)
            minErrors = np.min(labelsData[1])
            if labelsData[1][0] == minErrors: # less 0's
                indicesInLabels = np.where((labels == 0))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less zeros!")
                    print("associated id label:", indicesInLabels)              
                    print("associated X ids:", indicesX)
            elif labelsData[1][1] == minErrors: # less 1's
                indicesInLabels = np.where((labels == 1))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less ones!")
                    print("associated id label:", indicesInLabels)     
                    print("associated X ids:", indicesX)
            else:
                print("internal error, exiting")
                exit(-1)
            if verbose >= 2:
                print("min errors possible : ", minErrors)
            incorrCnt += minErrors
            representatives.append(representant)
            cardinalities.append(minErrors)
            #print("Representant = ", representant, ", min errors = ", minErrors)
        else:
            if verbose >= 2:
                print("no inconsistency")
    if verbose >= 1:
        print("At least ", incorrCnt, " elements can not be classified correctly.")
        print("accuracy upper bound = 1 - ", incorrCnt, "/", X.shape[0], " (", 1.0-(incorrCnt/X.shape[0]), ")")        
    return 1.0-(incorrCnt/X.shape[0])

#ALL from utils (you should modify it)
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


class Dataset():
    """Class representing a dataset loaded from a csv file.
    csv file is after rule mining"""
    def __init__(self, dataset_name, X, y, features, prediction):
        self.name = dataset_name
        self.X = X
        self.features = features
        self.y = y
        self.prediction = prediction
        self.preprocessed = False
        

    def data_modification(self, dataset_name, X, features):
        """modify dataset to have mutually exclusive demographic subgroups

        Args:
            dataset_name (str): name of the dataset
            X (numpy array): initial data read from csv
            features (list): list of features names corresponding to columns of X

        Returns:
            (numpy array , list ): (modified data, modified list of features)
        """
        df_x = pd.DataFrame(X, columns=features)
        if dataset_name == 'compas':
            df_x['Age=24-25'] = df_x.apply(lambda row: 1 if row['Age=18-25']==1 & row['Age=24-30']==1  else 0, axis=1)
            df_x['Age=18-23']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=18-25'], axis=1)
            df_x['Age=26-30']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=24-30'], axis=1)
            df_x['Age=26-29']= df_x.apply(lambda row: 0 if row['Age>=30']==1 & row['Age=26-30']==1 else row['Age=26-30'], axis=1)
        elif dataset_name == 'adult':
            df_x['race_other'] = ((1-df_x['neg_race_Amer-Indian-Eskimo'] ) | (1-df_x['neg_race_Other']))
            df_x['race_Asian'] = (1- df_x['neg_race_Asian-Pac-Islander'])
        elif dataset_name == 'acs_employ':
            df_x['American Indian and Alaska Native']= (1-df_x['neg_American Indian and Alaska Native tribes specified;or American Indian or Alaska Native,not specified and no other'])
            df_x['Two or More Races'] = 1 - df_x['neg_Two or More Races']
            df_x['Alaska Native alone'] = 1 - df_x['neg_Alaska Native alone']
            df_x['Native Hawaiian and Other Pacific Islander alone'] = 1 - df_x['neg_Native Hawaiian and Other Pacific Islander alone']
            df_x['Two or More Races'] = 1 - df_x['neg_Two or More Races']
        return np.array(df_x), df_x.columns.tolist()


    @classmethod
    def load_from_csv(cls, fname, dataset_name):
        """
        Load a dataset from a csv file. The csv file must contain n_samples+1 rows, each with n_features+1
        columns. The last column of each sample is its prediction class, and the first row of the file
        contains the feature names and prediction class name.
        attention :this function is not used anymore (replaced by from_csv)
        Parameters
        ----------
        fname : str
            File name of the csv data file
        
        Returns
        -------
        X : array-like, shape = [n_samples, n_features]
            The sample data

        y : array-line, shape = [n_samples]
            The target values for the sample data
        
        features : list
            A list of strings of length n_features. Specifies the names of each of the features.

        prediction_name : str
            The name of the prediction class
        """
        import csv
        features = []
        prediction_name = ""

        with open(fname, "r") as f:
            features = f.readline().strip().split(",")
            prediction_name = features[-1]
            features = features[0:-1]

        data = np.genfromtxt(fname, dtype=np.uint8, skip_header=1, delimiter=",")

        X = data[:, 0:-1]
        y = data[:, -1]
        return cls(dataset_name, X, y, features, prediction_name)

    
    @classmethod
    def from_csv(cls, fname, dataset_name ):
        """
        Load a dataset from a csv file. The csv file must contain n_samples+1 rows, each with n_features+1
        columns. The last column of each sample is its prediction class, and the first row of the file
        contains the feature names and prediction class name.
        
        Parameters
        ----------
        fname : str
            File name of the csv data file
        
        Returns
        -------
        X : array-like, shape = [n_samples, n_features]
            The sample data

        y : array-line, shape = [n_samples]
            The target values for the sample data
        
        features : list
            A list of strings of length n_features. Specifies the names of each of the features.

        prediction_name : str
            The name of the prediction class
        """
        df = pd.read_csv(fname)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

        features = df.columns[:-1].tolist()
        prediction = df.columns[-1]
        return cls(dataset_name, X, y, features, prediction)


    def train_test_split(self, train_proportion, random_state):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1.0 - train_proportion, shuffle=True,random_state=random_state)
        X_dict = {'train': X_train, 'test':X_test }
        y_dict = {'train': y_train, 'test':y_test }
        return X_dict, y_dict
    

    def pre_process (self):
        """This method apply all modifications regarding demographic subgroups
        """
        self.X, self.features = self.data_modification(self.name, self.X, self.features)
        self.preprocessed = True


    def get_data_norulemining(self, splits, random_state_param=42):
        """This method split data to train and test set after preprocessing

        Args:
            splits (dict): example {"train" : 0.8, "test" : 0.2}
            random_state_param (int, optional):  Defaults to 42.

        Returns:
            dict: the output is X={'train':X_train,'test': X_test}
        and y={'train':y_train,'test': y_test}
        """
        # Pre-process data to add demographic groups columns
        if not self.preprocessed:
            self.pre_process()

        # Generate splits
        assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
        split_names = list(splits.keys())
        split_ratios = list(splits.values())
        assert np.sum(split_ratios) == 1, "The split ratios must sum up to one"
        X_dict = {}
        y_dict = {}
        X_1, X_2, y_1, y_2 = train_test_split(self.X, self.y, train_size=split_ratios[0],
                                            shuffle=True, random_state=random_state_param)
        X_dict[split_names[0]] = X_1
        y_dict[split_names[0]] = y_1
        if len(splits) == 2:
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
        else:
            sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
            X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=sub_ratio,
                                            shuffle=True, random_state=random_state_param)
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
            X_dict[split_names[2]] = X_3
            y_dict[split_names[2]] = y_3
        return X_dict, y_dict, self.features, self.prediction
    

    def demographicGroup(self):
        """define demographic groups based on dataset name"""
        if self.name == 'adult':
            condition_gender = ['gender_Male', 'gender_Female']
            condition_age = ['age_low', 'age_middle', 'age_high']
            condition_race = ['race_White', 'race_Black', 'race_Asian', 'race_other']
        if self.name == 'compas':
            condition_gender = ['Gender=Male', 'neg_Gender=Male']
            condition_age = ['Age=18-25','Age=26-29','Age>=30']
            condition_race = ['Race=African-American', 'Race=Caucasian', 'Race=Hispanic', 'Race=Other']
        if self.name == 'acs_employ':
            condition_gender = ['neg_Female','Female' ]
            condition_age = ['age_low','age_medium', 'age_high' ]
            condition_race = ['White alone','Black or African American alone','Asian alone','Some Other Race alone',\
                            'Two or More Races','American Indian and Alaska Native','Native Hawaiian and Other Pacific Islander alone']
        return {'Age': condition_age,
                'Gender': condition_gender,
                'Race': condition_race,
                'All':condition_age+condition_gender+condition_race}
    

    def filter_subgroup_data(self,condition):
        """filter data for specific subgroup condition"""  
        filtered_data = self.X[:, [i for i,j in enumerate(self.features) if j in condition]]
        if len (condition)==1:
            filtered_data = filtered_data.ravel()
        return filtered_data


    def get_condition_freq(self, condition):
        condition_count = self.filter_subgroup_data(condition).sum(axis=0)
        freq = (condition_count/self.X.shape[0])*100
        return dict(zip(condition, freq))

    
    def corr_subgroups(self, condition):
        correlation= {}
        for i,j in enumerate(condition):
            corr = np.corrcoef(self.y,self.filter_subgroup_data(condition)[:,i])[0,1]
            correlation[j]=corr
        return correlation
    

    def to_df(self):
        return pd.DataFrame(self.X, columns=self.features)
    
    
    def to_df_from_dict(self, X_dict):
        df_X = {}
        for key, val in X_dict.items():
            df_X[key] = pd.DataFrame(val, columns=self.features)
        return df_X

    