import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import FairnessMeasure #change it's place to method class definition
from utils import age_data_modification, save_json

def parse_confusion_matrix(CM):
    """
    CM shape:
    CM[source][true_label, pred_label]
    source: interpretable, Blackbox
    """

    return {
        'T': {  # interpretable
            'TN': int(CM['Interpretable'][0, 0]),
            'FP': int(CM['Interpretable'][0, 1]),
            'FN': int(CM['Interpretable'][1, 0]),
            'TP': int(CM['Interpretable'][1, 1]),
        },
        'B': {  # black-box
            'TN': int(CM['Blackbox'][0, 0]),
            'FP': int(CM['Blackbox'][0, 1]),
            'FN': int(CM['Blackbox'][1, 0]),
            'TP': int(CM['Blackbox'][1, 1]),
        }
    }


def evaluate_group(
    X, y, preds, preds_type, condition, features):
    fairness = FairnessMeasure(X, features, [condition])

    icf = fairness.compute_fairness(preds_type, complement=False)['percentage_interpretable']

    CM = fairness.confusion_matrix(
        preds, y, preds_type,
        fairness.cond_indices,
        detailed=True
    )

    return icf, parse_confusion_matrix(CM)







dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"

# Load data using built-in method
X, y, features, prediction = load_from_csv("data/%s_mined.csv" %dataset_name)

#add extra age columns to the data
X, features = age_data_modification(X,features)

train_proportion = 0.8

condition_gender = ['Gender=Male', 'neg_Gender=Male']
condition_race = [i for i in features if 'Race' in i and '&&' not in i and 'neg' not in i] # all race subgropus in compas
condition_age = ['Age=18-25','Age=26-29','Age>=30']
#what if I add all the conditions here 
condition = condition_gender + condition_race + condition_age

# print(get_condition_freq(X, features,condition_gender ))
n_seeds = 5

data = {}
for min_coverage in [0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    data[min_coverage] = {}
    for Method in ['HybridCORELSPreClassifier' , 'HybridCORELSPostClassifier']:
        trans_total = {'train': {i:{'ICF':[],'FN':{'T':[],'B':[]}, 'FP':{'T':[],'B':[]},\
                             'TN':{'T':[],'B':[]}, 'TP':{'T':[],'B':[]}} for i in condition},\
            'test': {i:{'ICF':[],'FN':{'T':[],'B':[]}, 'FP':{'T':[],'B':[]},\
             'TN':{'T':[],'B':[]}, 'TP':{'T':[],'B':[]}}  for i in condition}}
        for i in range(n_seeds):
            np.random.seed(i)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True,random_state=i)

            # Set parameters
            corels_params = {'policy':"objective", 'max_card':1, 'n_iter':10**5, 'min_support':0.05, 'verbosity':["hybrid"]} # Add "progress" to verbosity to display detailed information about the search!
            alpha_value = 2 # Specialization Coefficient (see Section 3.1.2 of our paper)
            lambdaValue = 0.001 # Regularization coefficient for sparsity
            beta_value = min([ (1 / X_train.shape[0]) / 2, lambdaValue / 2]) # Regularization coefficient for transparency - this value ensures that transparency will break ties between identically accurate and sparse models
            #min_coverage = 0.30 # Desired minimum transparency (coverage of the interpretable part)

            t_limit = 60 # Seconds 
            m_limit = 4000 # MB

            

            # Define a hybrid model
            bbox = RandomForestClassifier(min_samples_split=10, max_depth=10,random_state=i)
            if Method == 'HybridCORELSPreClassifier':
                hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, 
                                                beta=beta_value, 
                                                c= lambdaValue, 
                                                alpha=alpha_value, 
                                                min_coverage=min_coverage, 
                                                obj_mode='collab', # 'collab' (recommended) matches the algorithm introduced in Section 4.4 of our paper, 'no_collab' is its variant proposed in the Appendix C
                                                **corels_params)#"progress"
            else:   
                hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox,  
                                                    bb_pretrained=False,
                                                    beta=beta_value, 
                                                    c= lambdaValue, 
                                                    min_coverage=min_coverage, 
                                                    **corels_params)#"progress"
        
            hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=t_limit, memory_limit=m_limit)

            #print("Status = ", hyb_model.get_status()) # Indicates whether the training was performed to optimality or if any other ending condition was reached

            #print("=> Trained model :", hyb_model)

            # Evaluate training performances
            preds_train, preds_types_train = hyb_model.predict_with_type(X_train)
            preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
            index_one_train = np.where(preds_types_counts_train[0] == 1)
            cover_rate_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])
            # print("=> Training accuracy = ", np.mean(preds_train == y_train))
            # print("=> Training transparency = ", cover_rate_train)

            # Evaluate test performances
            preds_test, preds_types_test = hyb_model.predict_with_type(X_test)
            preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
            index_one_test = np.where(preds_types_counts_test[0] == 1)
            cover_rate_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])
            # print("=> Test accuracy = ", np.mean(preds_test == y_test))
            # print("=> Test transparency = ", cover_rate_test)


            #compute fairness for each subgroup in condition
            for cond in condition:
                # #fairness_gender = FairnessMeasure(X_train_mod if 'Age' in cond else X_train, features_mod, [cond])
                # fairness = FairnessMeasure(X_train, features, [cond])
                # fairness_value = fairness.compute_fairness(preds_types_train, complement= False)['percentage_interpretable']
                # trans_total['train'][cond]['ICF'].append(fairness_value)
                # CM = fairness.confusion_matrix(preds_train, y_train, preds_types_train,\
                #                                 fairness.cond_indices, detailed = True)
                # trans_total['train'][cond]['FN']['T'].append(int(CM['Interpretable'][1,0])) #transparent, FN
                # trans_total['train'][cond]['FN']['B'].append(int(CM['Blackbox'][1,0])) #BB, FN
                # trans_total['train'][cond]['FP']['T'].append(int(CM['Interpretable'][0,1]))# transparent, FP
                # trans_total['train'][cond]['FP']['B'].append(int(CM['Blackbox'][0,1]))#BB, FP 
                # trans_total['train'][cond]['TN']['T'].append(int(CM['Interpretable'][0,0])) #transparent, TN
                # trans_total['train'][cond]['TN']['B'].append(int(CM['Blackbox'][0,0])) #BB, TN
                # trans_total['train'][cond]['TP']['T'].append(int(CM['Interpretable'][1,1]))# transparent, TP
                # trans_total['train'][cond]['TP']['B'].append(int(CM['Blackbox'][1,1]))#BB, TP
                # #for test
                # fairness = FairnessMeasure(X_test, features, [cond])
                # fairness_value = fairness.compute_fairness(preds_types_test, complement= False)['percentage_interpretable']
                # trans_total['test'][cond]['ICF'].append(fairness_value)
                # CM = fairness.confusion_matrix(preds_test, y_test, preds_types_test,\
                #                                     fairness.cond_indices, detailed = True)
                
                # trans_total['test'][cond]['FN']['T'].append(int(CM['Interpretable'][1,0])) #transparent, FN
                # trans_total['test'][cond]['FN']['B'].append(int(CM['Blackbox'][1,0])) #BB, FN
                # trans_total['test'][cond]['FP']['T'].append(int(CM['Interpretable'][0,1]))# transparent, FP
                # trans_total['test'][cond]['FP']['B'].append(int(CM['Blackbox'][0,1]))#BB, FP 
                # trans_total['test'][cond]['TN']['T'].append(int(CM['Interpretable'][0,0])) #transparent, TN
                # trans_total['test'][cond]['TN']['B'].append(int(CM['Blackbox'][0,0])) #BB, TN
                # trans_total['test'][cond]['TP']['T'].append(int(CM['Interpretable'][1,1]))# transparent, TP
                # trans_total['test'][cond]['TP']['B'].append(int(CM['Blackbox'][1,1]))#BB, TP

                # ---- TRAIN ----
                icf, cm = evaluate_group(
                    X_train, y_train,
                    preds_train, preds_types_train,
                    cond, features
                )

                trans_total['train'][cond]['ICF'].append(icf)

                for src in ['T', 'B']:
                    for k in ['TP', 'FP', 'TN', 'FN']:
                        trans_total['train'][cond][k][src].append(cm[src][k])

                # ---- TEST ----
                icf, cm = evaluate_group(
                    X_test, y_test,
                    preds_test, preds_types_test,
                    cond, features
                )

                trans_total['test'][cond]['ICF'].append(icf)
                
                for src in ['T', 'B']:
                    for k in ['TP', 'FP', 'TN', 'FN']:
                        trans_total['test'][cond][k][src].append(cm[src][k])
        data[min_coverage][Method]= trans_total

print(data)
#save_json(data,f"fairnesswithFNFPNEWALL.json")