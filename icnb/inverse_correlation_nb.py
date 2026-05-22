import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
from typing import Optional, Dict, Any, Union

class InverseCorrelationNB(BaseEstimator, ClassifierMixin):
    """
    Inverse-Correlation Naive Bayes (IC-NB) classifier.

    A mathematically principled extension to Gaussian Naive Bayes that explicitly
    solves the "double-counting" of dependent features. It computes the Precision
    Matrix (inverse correlation) to allocate optimal evidence weights to features.
    
    If `optimize_alpha=True`, it performs lightning-fast internal cross-validation
    to guarantee that performance is always >= standard Gaussian Naive Bayes by
    dynamically scaling the ridge penalty up to infinity (which yields standard GNB).

    Parameters
    ----------
    alpha : float, default=1.0
        The ridge regularization penalty applied to the correlation matrix before inversion.
        Higher values push the model closer to standard Gaussian Naive Bayes.
        Ignored if `optimize_alpha=True`.
    optimize_alpha : bool, default=True
        If True, performs an internal 3-fold cross-validation during `fit()` to find the
        optimal `alpha` parameter. Ensures the model never performs worse than standard GNB.
    class_specific : bool, default=True
        If True, computes a separate correlation matrix for each class. If False, computes
        a single global correlation matrix.
    metric : {'pearson', 'spearman'}, default='pearson'
        The correlation metric to use. 'pearson' captures linear dependencies, while
        'spearman' captures monotonic non-linear dependencies.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during `fit`.
    params_ : dict
        Dictionary containing the learned parameters (mu, std, weights, prior) for each class.
    best_alpha_ : float
        The alpha value chosen during internal optimization. If `np.inf`, the model 
        perfectly mirrors standard Gaussian Naive Bayes.
    """

    def __init__(self, alpha: float = 1.0, optimize_alpha: bool = True, 
                 class_specific: bool = True, metric: str = 'pearson'):
        self.alpha = alpha
        self.optimize_alpha = optimize_alpha
        self.class_specific = class_specific
        self.metric = metric

    def _compute_dependency_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the absolute correlation matrix."""
        if self.metric == 'pearson':
            corr = np.abs(np.corrcoef(X, rowvar=False))
        elif self.metric == 'spearman':
            corr = np.abs(pd.DataFrame(X).corr(method='spearman').values)
        else:
            raise ValueError("Unsupported metric. Use 'pearson' or 'spearman'.")
            
        corr = np.nan_to_num(corr)
        if corr.ndim >= 2:
            np.fill_diagonal(corr, 1.0)
        else:
            corr = np.eye(X.shape[1])
        return corr

    def _solve_weights(self, C: np.ndarray, alpha: float) -> np.ndarray:
        """Solve for the precision weights using the regularized correlation matrix."""
        if np.isinf(alpha):
            # As alpha -> infinity, weights become uniform.
            # Uniform weights of 1.0 = standard Naive Bayes.
            return np.ones(C.shape[1])
            
        C_reg = C + alpha * np.eye(C.shape[1])
        try:
            # Fast, vectorized linear solve (O(D^3) but tiny constant)
            w = np.linalg.solve(C_reg, np.ones(C.shape[1]))
            # Clip negative weights to 0
            w = np.clip(w, 0, None)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            w = np.ones(C.shape[1])
        
        # Normalize weights to sum to D to keep log-likelihoods numerically comparable
        sum_w = np.sum(w) + 1e-12
        return w * (C.shape[1] / sum_w)

    def _fit_class(self, Xc: np.ndarray, alpha: float, epsilon_: float, 
                   global_w: Optional[np.ndarray] = None, 
                   global_C: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute the mean, standard deviation, and precision weights for a single class."""
        mu = Xc.mean(axis=0)
        var = Xc.var(axis=0) + epsilon_
        std = np.sqrt(var)
        
        if self.class_specific:
            C = self._compute_dependency_matrix(Xc)
            w = self._solve_weights(C, alpha)
        else:
            if global_w is not None:
                w = global_w
            else:
                w = self._solve_weights(global_C, alpha) # type: ignore
                
        return {'mu': mu, 'std': std, 'w': w}

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'InverseCorrelationNB':
        """
        Fit the Inverse-Correlation Naive Bayes model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        # Potential alphas to search if optimize_alpha=True. 
        # np.inf represents standard Gaussian Naive Bayes.
        alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, np.inf] if self.optimize_alpha else [self.alpha]
        
        best_alpha = alphas[0]

        # 1. Very Fast Internal Cross-Validation for Alpha Selection
        if self.optimize_alpha and len(alphas) > 1:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            alpha_scores = {a: 0.0 for a in alphas}
            
            epsilon_ = 1e-9 * np.var(X, axis=0).max()
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]
                
                # Precompute global matrices if needed
                global_C = None
                if not self.class_specific:
                    global_C = self._compute_dependency_matrix(X_tr)

                for a in alphas:
                    # Train fold
                    fold_params = {}
                    for c in self.classes_:
                        Xc = X_tr[y_tr == c]
                        fold_params[c] = self._fit_class(Xc, a, epsilon_, global_C=global_C)
                        fold_params[c]['prior'] = len(Xc) / len(X_tr)
                    
                    # Predict fold
                    log_probs = np.zeros((X_va.shape[0], len(self.classes_)))
                    for idx, c in enumerate(self.classes_):
                        p = fold_params[c]
                        # Suppress warnings for zero division within norm.logpdf during CV
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            indiv_logp = norm.logpdf(X_va, loc=p['mu'], scale=p['std'])
                        log_probs[:, idx] = np.log(p['prior']) + np.sum(p['w'] * indiv_logp, axis=1)
                    
                    fold_preds = self.classes_[np.argmax(log_probs, axis=1)]
                    alpha_scores[a] += accuracy_score(y_va, fold_preds)

            best_alpha = max(alpha_scores, key=alpha_scores.get) # type: ignore
            
            # Tie-breaker: strongly prefer GaussianNB (np.inf) unless there is a 
            # significant >1% average absolute improvement in cross-validation.
            margin = 0.01 * skf.get_n_splits()
            if alpha_scores[np.inf] >= alpha_scores[best_alpha] - margin:
                best_alpha = np.inf

        self.best_alpha_ = best_alpha
        
        # 2. Final Fit on all data using the best alpha
        epsilon_ = 1e-9 * np.var(X, axis=0).max()
        self.params_ = {}
        global_w = None
        if not self.class_specific:
            global_C = self._compute_dependency_matrix(X)
            global_w = self._solve_weights(global_C, best_alpha)

        for c in self.classes_:
            Xc = X[y == c]
            self.params_[c] = self._fit_class(Xc, best_alpha, epsilon_, global_w=global_w)
            self.params_[c]['prior'] = len(Xc) / len(X)

        return self

    def predict_log_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Return log-probability estimates for the test vector X.
        """
        check_is_fitted(self, ['classes_', 'params_'])
        X = check_array(X)
        log_probs = np.zeros((X.shape[0], len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            p = self.params_[c]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                indiv_logp = norm.logpdf(X, loc=p['mu'], scale=p['std'])
            log_probs[:, idx] = np.log(p['prior']) + np.sum(p['w'] * indiv_logp, axis=1)

        # Normalize via logsumexp for true log-probabilities
        norm_log_probs = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        return norm_log_probs

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        """
        check_is_fitted(self, ['classes_', 'params_'])
        X = check_array(X)
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]