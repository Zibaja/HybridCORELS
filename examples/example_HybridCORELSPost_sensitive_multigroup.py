import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def compute_coverage_stats(pred_types, sensitive_matrix, group_names):
    """Return per-group coverage and max disparity (max-min)."""
    pred_types = np.asarray(pred_types)
    sensitive_matrix = np.asarray(sensitive_matrix)

    group_coverage = {}
    for idx, group_name in enumerate(group_names):
        group_mask = sensitive_matrix[:, idx].astype(bool)
        group_coverage[group_name] = float(np.mean(pred_types[group_mask] == 1))

    disparity = float(max(group_coverage.values()) - min(group_coverage.values()))
    return group_coverage, disparity


dataset_name = "compas"  # Supported: "compas", "adult", "acs_employ"

# Load data using built-in method
X, y, features, prediction = load_from_csv("data/%s_mined.csv" % dataset_name)

# Build a multi-group sensitive attribute matrix.
# Group 0 -> African-American
# Group 1 -> Hispanic
# Group 2 -> Other race categories
sensitive_aa_name = "Race=African-American"
sensitive_hispanic_name = "Race=Hispanic"
for sensitive_feature_name in [sensitive_aa_name, sensitive_hispanic_name]:
    if sensitive_feature_name not in features:
        raise ValueError("Missing expected sensitive feature: %s" % sensitive_feature_name)

sensitive_aa_idx = features.index(sensitive_aa_name)
sensitive_hispanic_idx = features.index(sensitive_hispanic_name)

sensitive_aa = X[:, sensitive_aa_idx].astype(np.uint8)
sensitive_hispanic = X[:, sensitive_hispanic_idx].astype(np.uint8)

# If the dataset is one-hot encoded as expected, each sample belongs to exactly one group.
sensitive_other = ((sensitive_aa == 0) & (sensitive_hispanic == 0)).astype(np.uint8)
sensitive = np.column_stack([sensitive_aa, sensitive_hispanic, sensitive_other])

# Generate train and test sets
random_state_param = 42
train_proportion = 0.8
(
    X_train,
    X_test,
    y_train,
    y_test,
    sensitive_train,
    sensitive_test,
) = train_test_split(
    X,
    y,
    sensitive,
    test_size=1.0 - train_proportion,
    shuffle=True,
    random_state=random_state_param,
)

# Set parameters
corels_params = {
    "policy": "objective",
    "max_card": 1,
    "n_iter": 10**7,
    "min_support": 0.05,
    "verbosity": ["hybrid"],
}
lambda_value = 0.001
beta_value = min([(1 / X_train.shape[0]) / 2, lambda_value / 2])
min_coverage = 0.8
max_coverage_disparity = 0.01

# Define a hybrid model (Post: BB first, interpretable part second)
bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)
hyb_model = HybridCORELSPostClassifier(
    black_box_classifier=bbox,
    bb_pretrained=False,
    beta=beta_value,
    c=lambda_value,
    min_coverage=min_coverage,
    max_coverage_disparity=max_coverage_disparity,
    **corels_params
)

# Train the hybrid model
t_limit = 60  # Seconds
m_limit = 4000  # MB
hyb_model.fit(
    X_train,
    y_train,
    sensitive_train=sensitive_train,
    features=features,
    prediction_name=prediction,
    time_limit=t_limit,
    memory_limit=m_limit,
)

print("Status =", hyb_model.get_status())
print("=> Trained model:", hyb_model)

# Evaluate training performances
preds_train, pred_types_train = hyb_model.predict_with_type(X_train)
cover_rate_train = float(np.mean(pred_types_train == 1))
train_group_cov, train_disparity = compute_coverage_stats(
    pred_types_train,
    sensitive_train,
    ["African-American", "Hispanic", "Other"],
)
print("=> Training accuracy =", float(np.mean(preds_train == y_train)))
print("=> Training transparency =", cover_rate_train)
print("=> Training group coverages =", train_group_cov)
print("=> Training coverage disparity =", train_disparity)

# Evaluate test performances
preds_test, pred_types_test = hyb_model.predict_with_type(X_test)
cover_rate_test = float(np.mean(pred_types_test == 1))
test_group_cov, test_disparity = compute_coverage_stats(
    pred_types_test,
    sensitive_test,
    ["African-American", "Hispanic", "Other"],
)
print("=> Test accuracy =", float(np.mean(preds_test == y_test)))
print("=> Test transparency =", cover_rate_test)
print("=> Test group coverages =", test_group_cov)
print("=> Test coverage disparity =", test_disparity)
