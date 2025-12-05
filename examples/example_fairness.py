import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import FairnessMeasure #change it's place to method class definition
from utils import age_data_modification, save_json





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
#condition = condition_age

trans_total = {'train': {i:[] for i in condition}, 'test': {i:[] for i in condition}}

#age_condition = ['Age=18-22','Age=24-30','Age>=30']
Method = 'HybridCORELSPostClassifier'  # Supported: "HybridCORELSPreClassifier", "HybridCORELSPostClassifier"


# print(get_condition_freq(X, features,condition_gender ))

for i in range(100):
    np.random.seed(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True,random_state=i)

    # Set parameters
    corels_params = {'policy':"objective", 'max_card':1, 'n_iter':10**5, 'min_support':0.05, 'verbosity':["hybrid"]} # Add "progress" to verbosity to display detailed information about the search!
    alpha_value = 2 # Specialization Coefficient (see Section 3.1.2 of our paper)
    lambdaValue = 0.001 # Regularization coefficient for sparsity
    beta_value = min([ (1 / X_train.shape[0]) / 2, lambdaValue / 2]) # Regularization coefficient for transparency - this value ensures that transparency will break ties between identically accurate and sparse models
    min_coverage = 0.80 # Desired minimum transparency (coverage of the interpretable part)

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

    print("=> Trained model :", hyb_model)

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

    #before computing fairness, create the modified datasets with mutually exclusive age groups
    # X_train_mod, features_mod = age_data_modification(X_train,features)
    # X_test_mod, features_mod = age_data_modification(X_test,features)

    #compute fairness for each subgroup in condition
    for cond in condition:
        #fairness_gender = FairnessMeasure(X_train_mod if 'Age' in cond else X_train, features_mod, [cond])
        fairness_gender = FairnessMeasure(X_train, features, [cond])
        fairness_value = fairness_gender.compute_fairness(preds_types_train, complement= False)['percentage_interpretable']
        trans_total['train'][cond].append(fairness_value)
        #for test
        fairness_gender = FairnessMeasure(X_test, features, [cond])
        fairness_value = fairness_gender.compute_fairness(preds_types_test, complement= False)['percentage_interpretable']
        trans_total['test'][cond].append(fairness_value)




save_json({Method: trans_total},f"fairness_0.8_{Method}.json")



