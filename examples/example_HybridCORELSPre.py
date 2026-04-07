import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"

# Load data using built-in method
X, y, features, prediction = load_from_csv("data/%s_mined.csv" %dataset_name)

# Generate train and test sets
random_state_param = 42
train_proportion = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True, random_state=random_state_param)

# Set parameters
corels_params = {'policy':"objective", 'max_card':1, 'n_iter':10**5, 'min_support':0.05, 'verbosity':["hybrid"]} # Add "progress" to verbosity to display detailed information about the search!
alpha_value = 2 # Specialization Coefficient (see Section 3.1.2 of our paper)
lambdaValue = 0.001 # Regularization coefficient for sparsity
beta_value = min([ (1 / X_train.shape[0]) / 2, lambdaValue / 2]) # Regularization coefficient for transparency - this value ensures that transparency will break ties between identically accurate and sparse models
min_coverage = 0.8 # Desired minimum transparency (coverage of the interpretable part)

# Define a hybrid model
bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)
hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, 
                                        beta=beta_value, 
                                        c= lambdaValue, 
                                        alpha=alpha_value, 
                                        min_coverage=min_coverage, 
                                        obj_mode='collab', # 'collab' (recommended) matches the algorithm introduced in Section 4.4 of our paper, 'no_collab' is its variant proposed in the Appendix C
                                        **corels_params)#"progress"

# Train the hybrid model
# Set resources used to train the prefix (interpretable part of the hybrid model)
t_limit = 60 # Seconds
m_limit = 4000 # MB
hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=t_limit, memory_limit=m_limit)

print("Status = ", hyb_model.get_status()) # Indicates whether the training was performed to optimality or if any other ending condition was reached

print("=> Trained model :", hyb_model)



print("test by ziba:")
print(hyb_model.interpretable_part.rl().rules)
for i in hyb_model.interpretable_part.rl().rules:
    print(features[(i['antecedents'][0]-1)])
print([i['antecedents'][0]-1 for i in hyb_model.interpretable_part.rl().rules][:-1])

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