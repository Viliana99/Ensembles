import numpy as np

from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from numpy.random import randint, permutation


class RandomForestMSE:

    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = 1/3 if feature_subsample_size is None else feature_subsample_size
        self.trees_parameters = trees_parameters
        self.regs = []
        self.idxs = []

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        for _ in range(self.n_estimators):
            idx = randint(0, X.shape[0], X.shape[0])
            X_new, y_new = X[idx], y[idx]
            idx = permutation(X.shape[1])[:int(self.feature_subsample_size * X.shape[1])]
            X_new = X_new[:,idx]
            self.idxs.append(idx)
            reg = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth)
            reg.fit(X_new, y_new)
            self.regs.append(reg)
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[0])
        for i,reg in enumerate(self.regs):
            pred = pred + reg.predict(X[:, self.idxs[i]])
        return pred / self.n_estimators
       
class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = 1/3 if feature_subsample_size is None else feature_subsample_size
        self.trees_parameters = trees_parameters
        self.loss = lambda x, y: ((x - y) ** 2).sum() / 2
        self.loss_grad = lambda x, y: x - y
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self.regs = []
        self.gammas = []
        self.idxs = []
        
        fit_preds = np.zeros(X.shape[0])
        for _ in range(self.n_estimators):
            s = - self.loss_grad(fit_preds, y)
            #idx_obj = randint(0, X.shape[0], X.shape[0])
            #X_new, y_new = X[idx_obj], y[idx_obj] 
            X_new, y_new = X, y
            idx_f = permutation(X.shape[1])[:int(self.feature_subsample_size * X.shape[1])]
            X_new = X_new[:,idx_f]
            self.idxs.append(idx_f)
            reg = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth)
            reg.fit(X_new, s)
            pred = reg.predict(X[:,idx_f]) 
            self.regs.append(reg)
            gamma =  minimize_scalar(lambda alpha: self.loss(y, fit_preds + alpha * pred))
            fit_preds = fit_preds +  self.learning_rate * gamma.x * pred
            self.gammas.append(gamma.x)
                
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y = y + self.gammas[i] * self.regs[i].predict(X[:, self.idxs[i]])
        return self.learning_rate * y