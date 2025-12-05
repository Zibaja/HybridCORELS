from .PrefixCorels import PrefixCorelsPreClassifier
from .PrefixCorels import PrefixCorelsPostClassifier
from .utils import check_consistent_length, check_array, get_feature, check_in, check_features
import numpy as np


class HybridCORELSPreClassifier:
    """Hybrid Rule List/Black-box based classifier.

    This class implements an Hybrid interpretable/black-box model.
    It uses a modified version of the CORELS algorithm (and its Python binding PyCORELS) to learn the intepretable part.
    This "interpretable part" consists in a prefix (ordered set of rules).
    Examples not determined by the prefix's rules are not classifier by a default prediction (as in traditional rule lists),
    but are rather used to train a black-box model, whose base class is user-defined.

    Attributes
    ----------
    c, n_iter, map_type, policy, verbosity, ablation, max_card, min_support : arguments of the CORELS algorithm 
    (see CORELS' documentation for details)

    obj_mode : str, optional (default:'collab')
        If 'collab', maximizes (prefix accuracy + BB accuracy UB) - i.e., takes care of the inconsistent examples let to the BB part (as in Section 4.4 of our paper).
        If 'no_collab', only maximizes the prefix's accuracy (as in the Appendix C of our paper)

    black_box_classifier: object for the black-box part of the interpretable model.

    alpha: black-box specialization coefficient (used to weight the black-box training set)

    verbosity: as in original CORELS, + "hybrid" to print information regarding the Hybrid model learning framework

    beta: float, regularization hyperparameter weighting transparency in the objective function. When using the min_coverage hard constraint,
        we recommend to set beta < 1/n_samples

    min_coverage: float (between 0.0 and 1.0), minimum acceptable value for the hybrid model transparency (proportion of examples classified by the interpretable part of the model)

    References
    ----------
    Original CORELS algorithm: Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin.
    Learning Certifiably Optimal Rule Lists for Categorical Data. KDD 2017.
    Journal of Machine Learning Research, 2018; 19: 1-77. arXiv:1704.01701, 2017
    """
    
    _estimator_type = "classifier"

    def __init__(self, black_box_classifier=None, c=0.001, n_iter=10**7, map_type="prefix", policy="lower_bound",
                 verbosity=["hybrid"], ablation=0, max_card=2, min_support=0.01, beta=0.0, alpha=0.0, min_coverage=0.0, random_state=42, obj_mode='collab'):
        # Retrieve parameters related to CORELS, and creation of the interpretable part of the Hybrid model
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.beta = beta
        self.alpha = alpha
        self.min_coverage=min_coverage
        self.obj_mode = obj_mode
        self.interpretable_part = PrefixCorelsPreClassifier(self.c, self.n_iter, self.map_type, self.policy, self.verbosity, self.ablation, self.max_card, self.min_support, self.beta, self.min_coverage, self.obj_mode)
        np.random.seed(random_state)
        # Creation of the black-box part of the Hybrid model
        if black_box_classifier is None:
            print("Unspecified black_box_classifier parameter, using sklearn RandomForestClassifier() for black-box part of the model.")
            from sklearn.ensemble import RandomForestClassifier
            black_box_classifier = RandomForestClassifier()
        self.BlackBoxClassifier = black_box_classifier
        self.black_box_part = self.BlackBoxClassifier

        # Done!
        self.is_fitted = False
        if "hybrid" in self.verbosity:
            print("Hybrid model created!")

    def load(fname):
        """
        Load a HybridCORELSPreClassifier from a file, using python's pickle module.
        
        Parameters
        ----------
        fname : string
            File name to load the rulelist from
        
        Returns
        -------
        self : obj
        """
        import pickle
        with open(fname, "rb") as f:
            loaded_object = pickle.load(f)
        if type(loaded_object) != HybridCORELSPreClassifier:
            raise TypeError("Loaded object of type %s from file %s, expected <class 'HybridCORELS.HybridCORELS.HybridCORELSPreClassifier'>" %(type(loaded_object), fname))
        else:
            return loaded_object

    def fit(self, X, y, features=[], prediction_name="prediction", specialization_auto_tuning=False, time_limit = None, memory_limit=None):
        """
        Build a CORELS classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.

        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        features : list, optional(default=[])
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].

        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.

        time_limit : int, maximum number of seconds allowed for the model building 
        (this timeout considers only the interpretable part building using the modified CORELS algorithm)
        Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

        memory_limit: int, maximum memory use (in MB)
        (this memory limit considers only the interpretable part building using the modified CORELS algorithm)

        specialization_auto_tuning: not implemented yet, should not be used
        
        Returns
        -------
        self : obj
        """
        # 1) Fit the interpretable part of the Hybrid model
        if "hybrid" in self.verbosity:
            print("Fitting the interpretable part...")
        self.interpretable_part.fit(X, y, features, prediction_name, time_limit=time_limit, memory_limit=memory_limit)

        # 2) Fit the black-box part of the Hybrid model (using examples not determined by the interpretable part)
        # Retrieve only examples not captured by the interpretable part
        interpretable_predictions = self.interpretable_part.predict(X)
        not_captured_indices = np.where(interpretable_predictions == 2)
        captured_indices = np.where(interpretable_predictions < 2)
        if "hybrid" in self.verbosity:
            print("Interpretable part coverage = ", (y.size-not_captured_indices[0].size)/y.size)
            print("Interpretable part accuracy = ", np.mean(interpretable_predictions[captured_indices] == y[captured_indices]))
            print("Fitting the black-box part on examples not captured by the interpretable part...")

        # Old way: fit black-box only on uncaptured examples only
        X_not_captured = X[not_captured_indices]
        y_not_captured = y[not_captured_indices]
        #self.black_box_part.fit(X_not_captured, y_not_captured)

        if not specialization_auto_tuning:
            # New way: weight training examples
            # First compute and set the sample weights
            examples_weights = np.ones(y.shape) # start from the uniform distribution
            examples_weights[not_captured_indices] = np.exp(self.alpha)
            examples_weights /= np.sum(examples_weights)
            #print("min(examples_weights) = ", np.min(examples_weights))
            #print("max(examples_weights) = ", np.max(examples_weights))
            # Then train the black-box on its weighted training set
            self.black_box_part.fit(X, y, sample_weight=examples_weights)
        else:
            print("Not implemented yet")
        '''else:
            alpha_list = np.arange(0,15,1)
            np.random.randint(y.shape, replace='False')
            for alpha_value in alpha_list:
                # First compute and set the sample weights
                examples_weights = np.ones(y.shape) # start from the uniform distribution
                examples_weights[not_captured_indices] = np.exp(self.alpha)
                examples_weights /= np.sum(examples_weights)

                # Then train the black-box on its weighted training set
                self.black_box_part.fit(X, y, sample_weight=examples_weights)'''
                
        # Finally set the black-box metrics
        self.black_box_support = not_captured_indices[0].size # Proportion of training examples falling into the black-box part
        if not_captured_indices[0].size > 0:
            self.black_box_accuracy = self.black_box_part.score(X_not_captured, y_not_captured) # Black-Box accuracy on these examples
            y_not_captured_unique_counts = np.unique(y_not_captured, return_counts=True)[1]
            self.black_box_majority = max(y_not_captured_unique_counts)/sum(y_not_captured_unique_counts)
            if "hybrid" in self.verbosity:
                #print("majority pred = ", self.black_box_majority, "BB accuracy = ", self.black_box_accuracy)
                print("Black-Box part accuracy = ", self.black_box_accuracy)
        else:
            self.black_box_accuracy = 1.00            
        # Done!
        self.is_fitted = True

        return self

    def refit_black_box(self, X, y, alpha, black_box_classifier):
        """
        Can be used to replace/retrain the black-box part of an HybridModel (pre-paradigm) without retraining the interpretable part

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.

        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        alpha : float or int, new specialization coefficient value

        black_box_classifier: classifier to be used as the black-box (will be trained)
        Returns
        -------
        self : obj
        """

        self.alpha = alpha
        self.black_box_part = black_box_classifier

        # 2) Fit the black-box part of the Hybrid model (using examples not determined by the interpretable part)
        # Retrieve only examples not captured by the interpretable part
        interpretable_predictions = self.interpretable_part.predict(X)
        not_captured_indices = np.where(interpretable_predictions == 2)
        captured_indices = np.where(interpretable_predictions < 2)
        #if "hybrid" in self.verbosity:
        #    print("Interpretable part coverage = ", (y.size-not_captured_indices[0].size)/y.size)
        #    print("Interpretable part accuracy = ", np.mean(interpretable_predictions[captured_indices] == y[captured_indices]))
        #    print("Fitting the black-box part on examples not captured by the interpretable part...")

        # Old way: fit black-box only on uncaptured examples only
        X_not_captured = X[not_captured_indices]
        y_not_captured = y[not_captured_indices]
        #self.black_box_part.fit(X_not_captured, y_not_captured)

        # New way: weight training examples
        # First compute and set the sample weights
        examples_weights = np.ones(y.shape) # start from the uniform distribution
        examples_weights[not_captured_indices] = np.exp(self.alpha)
        examples_weights /= np.sum(examples_weights)
        #print("min(examples_weights) = ", np.min(examples_weights))
        #print("max(examples_weights) = ", np.max(examples_weights))
        # Then train the black-box on its weighted training set
        self.black_box_part.fit(X, y, sample_weight=examples_weights)
        
        # Finally set the black-box metrics
        self.black_box_support = not_captured_indices[0].size # Proportion of training examples falling into the black-box part
        if not_captured_indices[0].size > 0:
            self.black_box_accuracy = self.black_box_part.score(X_not_captured, y_not_captured) # Black-Box accuracy on these examples
            y_not_captured_unique_counts = np.unique(y_not_captured, return_counts=True)[1]
            self.black_box_majority = max(y_not_captured_unique_counts)/sum(y_not_captured_unique_counts)
            if "hybrid" in self.verbosity:
                #print("majority pred = ", self.black_box_majority, "BB accuracy = ", self.black_box_accuracy)
                print("Black-Box part accuracy = ", self.black_box_accuracy)
        else:
            self.black_box_majority = 0.00 # arbitrary
            self.black_box_accuracy = 0.00 # arbitrary            
        # Done!
        self.is_fitted = True

        return self
    
    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)          
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]

        # Return overall prediction
        return overall_predictions

    def predict_proba(self, X):
        """
        Predict classification probabilities of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications probabilities of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict_proba(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict_proba(X)
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]
        
        # Return overall prediction
        return overall_predictions

    def predict_with_type(self, X):
        """
        Predict classifications of the input samples X, along with a boolean (one per example)
        indicating whether the example was classified by the interpretable part of the model or not.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p, t : array of shape = [n_samples], array of shape = [n_samples].
            p: The classifications of the input samples
            t: The part of the Hybrid model which decided for the classification (1: interpretable part, 0: black-box part).
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        # Craft the predictions type vector
        predictions_type = np.ones(shape=overall_predictions.shape)

        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)        
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]    
            predictions_type[not_captured_indices] = 0

         # Return overall prediction along with the part that classified the example (1: interpretable part, 0: black-box)
        return overall_predictions, predictions_type

    def __str__(self):
        s = "HybridCORELSPreClassifier"
        if not hasattr(self, 'black_box_majority'): # compatibility with old version
            self.black_box_majority = 0.00
            self.black_box_accuracy = 0.00
        if self.is_fitted:
            s += "\n" + self.interpretable_part.rl().__str__()
            #s += "\n    default: " + str(self.black_box_part) + "(support %d, accuracy %.5f (majority pred %.3f))" %(self.black_box_support, self.black_box_accuracy, self.black_box_majority)
            s += "\n    default: " + str(self.black_box_part) + "(support %d, accuracy %.5f)" %(self.black_box_support, self.black_box_accuracy)        
        else:
            s += "Not Fitted Yet!"
            
        return s

    def score(self, X, y):
        """
        Score the algorithm on the input samples X with the labels y. Alternatively,
        score the predictions X against the labels y (where X has been generated by 
        `predict` or something similar).

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
            The input samples, or the sample predictions. All features must be binary.
        
        y : array-like, shape = [n_samples]
            The input labels. All labels must be binary.

        Returns
        -------
        a : float
            The accuracy, from 0.0 to 1.0, of the rulelist predictions
        """

        labels = check_array(y, ndim=1)
        p = check_array(X)
        check_consistent_length(p, labels)
        
        if p.ndim == 2:
            p = self.predict(p)
        elif p.ndim != 1:
            raise ValueError("Input samples must have only 1 or 2 dimensions, got " + str(p.ndim) +
                             " dimensions")

        a = np.mean(np.invert(np.logical_xor(p, labels)))

        return a

    def get_sparsity(self):
        return len(self.interpretable_part.rl_.rules)-1

    def get_status(self):
        return self.interpretable_part.get_status()

    def save(self, fname):
        """
        Save the model to a file, using python's pickle module.

        Parameters
        ----------
        fname : string
            File name to store the model in
        
        Returns
        -------
        self : obj
        """
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(self, f)

        return self

class HybridCORELSPostClassifier:
    """Hybrid Rule List/Black-box based classifier.

    This class implements an Hybrid interpretable/black-box model.
    It uses a modified version of the CORELS algorithm (and its Python binding PyCORELS) to learn the intepretable part.
    This "interpretable part" consists in a prefix (ordered set of rules).
    Examples not determined by the prefix's rules are not classifier by a default prediction (as in traditional rule lists),
    but are rather used to train a black-box model, whose base class is user-defined.

    Attributes
    ----------
    c, n_iter, map_type, policy, verbosity, ablation, max_card, min_support : arguments of the CORELS algorithm 
    (see CORELS' documentation for details)

    bb_pretrained : boolean, optional (default=False)
        Indicates whether the given black-box is already trained or not.
        If False, the BB is trained here.
        If True, we check whether it is effectively fitted.

    black_box_classifier: object for the black-box part of the interpretable model.

    verbosity: as in original CORELS, + "hybrid" to print information regarding the Hybrid model learning framework

    beta: float, regularization hyperparameter weighting transparency in the objective function. When using the min_coverage hard constraint,
        we recommend to set beta < 1/n_samples

    min_coverage: float (between 0.0 and 1.0), minimum acceptable value for the hybrid model transparency (proportion of examples classified by the interpretable part of the model)

    References
    ----------
    Original CORELS algorithm: Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin.
    Learning Certifiably Optimal Rule Lists for Categorical Data. KDD 2017.
    Journal of Machine Learning Research, 2018; 19: 1-77. arXiv:1704.01701, 2017
    """
    
    _estimator_type = "classifier"

    def __init__(self, black_box_classifier=None, c=0.001, n_iter=10**7, map_type="prefix", policy="lower_bound",
                 verbosity=["hybrid"], ablation=0, max_card=2, min_support=0.01, beta=0.0, min_coverage=0.0, random_state=42, bb_pretrained=False):
        # Retrieve parameters related to CORELS, and creation of the interpretable part of the Hybrid model
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.beta = beta
        self.min_coverage=min_coverage
        self.bb_pretrained=bb_pretrained
        self.interpretable_part = PrefixCorelsPostClassifier(self.c, self.n_iter, self.map_type, self.policy, self.verbosity, self.ablation, self.max_card, self.min_support, self.beta, self.min_coverage)
        np.random.seed(random_state)
        
        # Creation of the black-box part of the Hybrid model
        if black_box_classifier is None:
            if self.bb_pretrained:
                raise ValueError("Parameters indicate that the black-box is pretrained but it is not provided!")
            print("Unspecified black_box_classifier parameter, using sklearn RandomForestClassifier() for black-box part of the model.")
            from sklearn.ensemble import RandomForestClassifier
            black_box_classifier = RandomForestClassifier()
        self.BlackBoxClassifier = black_box_classifier
        self.black_box_part = self.BlackBoxClassifier

        # If parameters indicate that BB is pretrained, verify it now
        if self.bb_pretrained:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.black_box_part)

        # Done!
        self.is_fitted = False
        if "hybrid" in self.verbosity:
            print("Hybrid model created!") 

    def load(fname):
        """
        Load a HybridCORELSPostClassifier from a file, using python's pickle module.
        
        Parameters
        ----------
        fname : string
            File name to load the rulelist from
        
        Returns
        -------
        self : obj
        """
        import pickle
        with open(fname, "rb") as f:
            loaded_object = pickle.load(f)
        if type(loaded_object) != HybridCORELSPostClassifier:
            raise TypeError("Loaded object of type %s from file %s, expected <class 'HybridCORELS.HybridCORELS.HybridCORELSPostClassifier'>" %(type(loaded_object), fname))
        else:
            return loaded_object

    def fit(self, X, y, features=[], prediction_name="prediction", time_limit = None, memory_limit=None):
        """
        Build a CORELS classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.

        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        features : list, optional(default=[])
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].

        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.

        time_limit : int, maximum number of seconds allowed for the model building 
        (this timeout considers only the interpretable part building using the modified CORELS algorithm)
        Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

        memory_limit: int, maximum memory use (in MB)
        (this memory limit considers only the interpretable part building using the modified CORELS algorithm)

        Returns
        -------
        self : obj
        """
        # 1) (if not pretrained) Fit the black-box part of the Hybrid model
        if not self.bb_pretrained:
            if "hybrid" in self.verbosity:
                print("Training the BB part on the entire dataset")
            self.black_box_part.fit(X, y)
        else:
            if "hybrid" in self.verbosity:
                print("Not retraining BB.")
        bb_errors = np.asarray(self.black_box_part.predict(X) != y).astype(int)
        #print("python computed bb error rate = ", np.mean(bb_errors))
        # 2) Fit the interpretable part of the model
        if "hybrid" in self.verbosity:
            print("Fitting the interpretable part...")
        self.interpretable_part.fit(X, y, bb_errors, features, prediction_name, time_limit=time_limit, memory_limit=memory_limit)

        interpretable_predictions = self.interpretable_part.predict(X)
        not_captured_indices = np.where(interpretable_predictions == 2)
        captured_indices = np.where(interpretable_predictions < 2)
        if "hybrid" in self.verbosity:
            print("Interpretable part coverage = ", (y.size-not_captured_indices[0].size)/y.size)
            print("Interpretable part accuracy = ", np.mean(interpretable_predictions[captured_indices] == y[captured_indices]))

        # Finally set the black-box metrics
        X_not_captured = X[not_captured_indices]
        y_not_captured = y[not_captured_indices]
        self.black_box_support = not_captured_indices[0].size # Proportion of training examples falling into the black-box part
        if not_captured_indices[0].size > 0:
            self.black_box_accuracy = self.black_box_part.score(X_not_captured, y_not_captured) # Black-Box accuracy on these examples
            y_not_captured_unique_counts = np.unique(y_not_captured, return_counts=True)[1]
            self.black_box_majority = max(y_not_captured_unique_counts)/sum(y_not_captured_unique_counts)
            if "hybrid" in self.verbosity:
                #print("majority pred = ", self.black_box_majority, "BB accuracy = ", self.black_box_accuracy)
                print("Black-Box part accuracy = ", self.black_box_accuracy)
        else:
            self.black_box_majority = 0.00 # arbitrary
            self.black_box_accuracy = 0.00 # arbitrary           

        # Done!
        self.is_fitted = True
        if "hybrid" in self.verbosity:
            print("Training accuracy overall = ", self.score(X, y))

        return self

    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)          
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]

        # Return overall prediction
        return overall_predictions

    def predict_proba(self, X):
        """
        Predict classification probabilities of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications probabilities of the input samples.
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict_proba(X)
        overall_predictions = interpretable_predictions
        
        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict_proba(X)
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]
        
        # Return overall prediction
        return overall_predictions

    def predict_with_type(self, X):
        """
        Predict classifications of the input samples X, along with a boolean (one per example)
        indicating whether the example was classified by the interpretable part of the model or not.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p, t : array of shape = [n_samples], array of shape = [n_samples].
            p: The classifications of the input samples
            t: The part of the Hybrid model which decided for the classification (1: interpretable part, 0: black-box part).
        """
        # Predict using the interpretable part of the Hybrid model
        interpretable_predictions = self.interpretable_part.predict(X)
        overall_predictions = interpretable_predictions
        # Craft the predictions type vector
        predictions_type = np.ones(shape=overall_predictions.shape)

        # Predict using the black-box part of the Hybrid model
        not_captured_indices = np.where(interpretable_predictions == 2)
        
        if not_captured_indices[0].size > 0:
            black_box_predictions = self.black_box_part.predict(X)        
            overall_predictions[not_captured_indices] = black_box_predictions[not_captured_indices]    
            predictions_type[not_captured_indices] = 0

         # Return overall prediction along with the part that classified the example (1: interpretable part, 0: black-box)
        return overall_predictions, predictions_type

    def __str__(self):
        s = "HybridCORELSPostClassifier"
        if not hasattr(self, 'black_box_majority'): # compatibility with old version
            self.black_box_majority = 0.00
            self.black_box_accuracy = 0.00
        if self.is_fitted:
            s += "\n" + self.interpretable_part.rl().__str__()
            #s += "\n    default: " + str(self.black_box_part) + "(support %d, accuracy %.3f (majority pred %.3f))" %(self.black_box_support, self.black_box_accuracy, self.black_box_majority)
            s += "\n    default: " + str(self.black_box_part) + "(support %d, accuracy %.3f)" %(self.black_box_support, self.black_box_accuracy)        
        else:
            s += "Not Fitted Yet!"
            
        return s

    def score(self, X, y):
        """
        Score the algorithm on the input samples X with the labels y. Alternatively,
        score the predictions X against the labels y (where X has been generated by 
        `predict` or something similar).

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
            The input samples, or the sample predictions. All features must be binary.
        
        y : array-like, shape = [n_samples]
            The input labels. All labels must be binary.

        Returns
        -------
        a : float
            The accuracy, from 0.0 to 1.0, of the rulelist predictions
        """

        labels = check_array(y, ndim=1)
        p = check_array(X)
        check_consistent_length(p, labels)
        
        if p.ndim == 2:
            p = self.predict(p)
        elif p.ndim != 1:
            raise ValueError("Input samples must have only 1 or 2 dimensions, got " + str(p.ndim) +
                             " dimensions")

        a = np.mean(np.invert(np.logical_xor(p, labels)))

        return a

    def get_sparsity(self):
        return len(self.interpretable_part.rl_.rules)-1

    def get_status(self):
        return self.interpretable_part.get_status()

    def save(self, fname):
        """
        Save the model to a file, using python's pickle module.

        Parameters
        ----------
        fname : string
            File name to store the model in
        
        Returns
        -------
        self : obj
        """
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(self, f)

        return self
    

    # Ziba tried to implement fairness within HybridCORELS but this only changes local file , so check with Julien
    # def fairness(self, X, features=[], condition=[]):
    #     if not features or not condition:
    #         raise ValueError("Features and condition parameters must be specified to compute fairness metrics.")
        
    #     def set_condition(condition):  
    #         indices_of_features = [i for i,j in enumerate(features) if j in condition]
    #         condition_indices = np.all((X[:,indices_of_features]),axis=1)
    #         return condition_indices
        
        
    #     def compute_fairness(preds_types,complement = False):
    #         condition_indices = set_condition(condition)
    #         if complement:
    #             condition_indices = np.logical_not(condition_indices)
    #         total_count = condition_indices.sum() # total number of samples satisfying the condition
    #         interpretable_count = preds_types[condition_indices].sum() # number of samples satisfying the condition going through the interpretable part
    #         percentage_interpretable = (interpretable_count / total_count) * 100 if total_count > 0 else 0
    #         return {
    #             "condition": self.condition,
    #             "total_count": int(total_count),
    #             "interpretable_count": int(interpretable_count),
    #             "percentage_interpretable": '%.2f' % percentage_interpretable
    #         }
    #     _, preds_types = self.predict_with_type(X)
    #     compute_fairness(preds_types)
    #     return self

