import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import FairnessMeasure

if __name__ == "__main__":

    dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"

    # Load data using built-in method
    X, y, features, prediction = load_from_csv("data/%s_mined.csv" %dataset_name)


    ### new section by ziba only for evulation of age groups
    #ofcourse by doing this, the initial mined rules are changed and so the results may differ
    #extract useful columns for age groups:
    #this is the modified Dataset 
    # df_x = pd.DataFrame(X, columns=features)
    # df_x['Age=24-25'] = df_x.apply(lambda row: 1 if row['Age=18-25']==1 & row['Age=24-30']==1  else 0, axis=1)
    # df_x['Age=18-23']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=18-25'], axis=1)
    # df_x['Age=26-30']= df_x.apply(lambda row: 0 if row['Age=24-25']==1 else row['Age=24-30'], axis=1)
    # df_x['Age=26-29']= df_x.apply(lambda row: 0 if row['Age>=30']==1 & row['Age=26-30']==1 else row['Age=26-30'], axis=1)
    # X = np.array(df_x)
    # features = df_x.columns.tolist()


    # Generate train and test sets
    random_state_param = 42
    train_proportion = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True, random_state=random_state_param)

    # Set parameters
    corels_params = {'policy':"objective", 'max_card':1, 'n_iter':10**5, 'min_support':0.05, 'verbosity':["hybrid"]} # Add "progress" to verbosity to display detailed information about the search!
    alpha_value = 2 # Specialization Coefficient (see Section 3.1.2 of our paper)
    lambdaValue = 0.001 # Regularization coefficient for sparsity
    beta_value = min([ (1 / X_train.shape[0]) / 2, lambdaValue / 2]) # Regularization coefficient for transparency - this value ensures that transparency will break ties between identically accurate and sparse models
    min_coverage = 0.90 # Desired minimum transparency (coverage of the interpretable part)

    # Define a hybrid model
    bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)

    hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox,  
                                            bb_pretrained=False,
                                            beta=beta_value, 
                                            c= lambdaValue, 
                                            min_coverage=min_coverage, 
                                            **corels_params)#"progress"

    # Train the hybrid model
    # Set resources used to train the prefix (interpretable part of the hybrid model)
    t_limit = 60 # Seconds
    m_limit = 4000 # MB
    hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=t_limit, memory_limit=m_limit)

    print("Status = ", hyb_model.get_status()) # Indicates whether the training was performed to optimality or if any other ending condition was reached

    print("=> Trained model :", hyb_model)

    # Evaluate training performances
    preds_train, preds_types_train = hyb_model.predict_with_type(X_train)
    preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
    index_one_train = np.where(preds_types_counts_train[0] == 1)
    cover_rate_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])
    print("=> Training accuracy = ", np.mean(preds_train == y_train))
    print("=> Training transparency = ", cover_rate_train)

    # Evaluate test performances
    preds_test, preds_types_test = hyb_model.predict_with_type(X_test)
    preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
    index_one_test = np.where(preds_types_counts_test[0] == 1)
    cover_rate_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])
    print("=> Test accuracy = ", np.mean(preds_test == y_test))
    print("=> Test transparency = ", cover_rate_test)


# test save / load with pickle
#hyb_model.save("test_save_load") # to save
#hyb_model = HybridCORELSPreClassifier.load("test_save_load") # to load

# to try out another black-box
#hyb_model.refit_black_box(X_train, y_train, alpha_value,  bbox)


#Example usage:
if __name__ == "__main__":
    fairness_gender = FairnessMeasure(X_train, features, condition=['Gender=Male'])
    print(f"The frequency of {fairness_gender.condition} is {fairness_gender.get_condition_freq():.2f} percent")
    fairness_result = fairness_gender.compute_fairness(preds_types_train, complement= False)
    print(f"Fairness results for {fairness_gender.condition} is: {fairness_result}")
    fairness_gender.confusion_matrix(preds_train,y_train, preds_types_train,fairness_gender.cond_indices, detailed=False  )

    print(f"The frequency of negation of {fairness_gender.condition} is {(100- fairness_gender.get_condition_freq()):.2f} percent")
    fairness_result = fairness_gender.compute_fairness(preds_types_train, complement= True)
    print(f"Fairness results for negation of {fairness_gender.condition} is: {fairness_result}")





