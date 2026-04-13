
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import get_data_norulemining, to_df, FairnessMeasure
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL

import time

"""This script is to check other method like HyRS and CRL
"""


random_state_param = 42
X, y, features, _ = get_data_norulemining("compas", {"train" : 0.8, "test" : 0.2})



print("--------------- BB ---------------\n")
df_X = to_df(X, features)
# Fit a black-box
bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
bbox.fit(df_X["train"], y["train"])
# Test performance
print("BB Accuracy : ", np.mean(bbox.predict(df_X["test"]) == y["test"]), "\n")



print("--------------- HyRS ---------------\n")
# Set parameters
hparams = {
    "alpha" : 0.001,
    "beta" : 0.1 #this significantly affects the coverage
}

# Define a hybrid model
hyb_model = HybridRuleSetClassifier(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(df_X["train"], y["train"], 10**7, T0=0.01, premined_rules=True, #50000
                                            random_state=3, time_limit=10)

print(hyb_model.get_description(df_X["test"], y["test"]))

print([features[i] for i in hyb_model.positive_rule_set])


preds_train, preds_types_train = hyb_model.predict_with_type(df_X["train"])
print(f"Train Coverage is {np.mean (preds_types_train)}")

preds_test, preds_types_test = hyb_model.predict_with_type(df_X["test"])
print(f"Test Coverage is {np.mean (preds_types_test)}")
print(X["train"].shape)
# fairness = FairnessMeasure(X["train"], features, ['Gender=Male'])
# fairness_value = fairness.compute_fairness(preds_types_train, complement= False)['percentage_interpretable']
# print(fairness_value)
# fairness = FairnessMeasure(X["train"], features, ['Gender=Male'])
# fairness_value = fairness.compute_fairness(preds_types_train, complement= True)['percentage_interpretable']
# print(fairness_value)
# fairness = FairnessMeasure(X["train"], features, ['neg_Gender=Male'])
# fairness_value = fairness.compute_fairness(preds_types_train, complement= False)['percentage_interpretable']
# print(fairness_value)



print("--------------- CRL ---------------\n")
start_time = time.perf_counter()
# Set parameters
hparams = {
    "max_card" : 2,
    "alpha" : 0.01
}

# Define a hybrid model
hyb_model = CRL(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(df_X["train"], y["train"], n_iteration=50000,random_state=random_state_param+1, 
                                                            premined_rules=True, time_limit=300)
print(hyb_model.get_description(df_X["test"], y["test"]))


y_pred, pred_type = hyb_model.predict_with_type(df_X["test"])


print("Black-box usage:", np.mean(pred_type == 0))
print("number of points sent to BB:", np.sum(pred_type == 0), 'All points:', len(pred_type))
print("Rule usage:", np.mean(pred_type == 1))
print("number of points sent to rulelist:", np.sum(pred_type == 1),'All points:', len(pred_type))

print(hyb_model.get_description(df_X["train"], y["train"]))

y_pred, pred_type = hyb_model.predict_with_type(df_X["train"])

print("For train dataset")
print("Black-box usage:", np.mean(pred_type == 0))
print("number of points sent to BB:", np.sum(pred_type == 0), 'All points:', len(pred_type))
print("Rule usage:", np.mean(pred_type == 1))
print("number of points sent to rulelist:", np.sum(pred_type == 1),'All points:', len(pred_type))

print(f'overall train accuracy {np.mean(y_pred == y["train"])}')
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Executed in {duration:.4f} seconds")
print(50*'-')

output_rules, rule_coverage, acc = hyb_model.test(df_X["train"], y["train"])
print(output_rules)
print(rule_coverage)
print(acc)


output_rules, rule_coverage, acc= hyb_model.test(df_X["test"], y["test"])
print(output_rules)
print('test rule coverage:')
print(rule_coverage)
print('test accuracy:')
print(acc)

# X_test is your test set
all_preds, all_types = hyb_model.predict_with_type_all(df_X["test"])

results = {"accuracy": {"train":[],"test":[]},"coverage":{"train":[],"test":[]}}
for i, (y_pred, pred_type) in enumerate(zip(all_preds, all_types)):
    acc = np.mean(y_pred == y["test"])
    coverage = np.mean(pred_type)
    results["accuracy"]["test"].append(acc)
    results["coverage"]["test"].append(coverage)
    print(f"Hybrid model with first {i+1} rules: ACC={acc:.3f}")

print(results)






# print("---------------  HybridCORELSPreClassifier ---------------")
# # Set parameters
# corels_params = {
#     'policy' : "objective", 
#     'max_card' : 1, 
#     'c' : 0.001, 
#     'n_iter' : 10**6,
#     'verbosity': []
# }

# alpha_value = 10
# beta_value = 0.0
# # Define a hybrid model
# hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, 
#                                       min_coverage=0.8, **corels_params)
# # Train the hybrid model
# hyb_model.fit(X["train"], y["train"], features=features)

# # Print the RuleList
# print("\n", hyb_model, "\n")

# # Test performance
# yhat, covered_index = hyb_model.predict_with_type(X["test"])
# print("HybridCORELSPreClassifier Accuracy : ", np.mean(yhat == y["test"])) 
# print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")



# print("---------------  HybridCORELSPostClassifier ---------------")

# hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=0.8, 
#                                        bb_pretrained=False, **corels_params)
# # Train the hybrid model
# hyb_model.fit(X["train"], y["train"], features=features)

# # Print the RuleList
# print("\n", hyb_model, "\n")

# # Test performance
# yhat, covered_index = hyb_model.predict_with_type(X["test"])
# print("HybridCORELSPostClassifier Accuracy : ", np.mean(yhat == y["test"])) 
# print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")