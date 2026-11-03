"""The model from the original notebook, plus the pieces that change it.

Base pipeline (as in the notebook): median imputation -> variance filter ->
SelectKBest(f_classif) -> RandomForest(class_weight="balanced"), tuned by
GridSearchCV. Everything here is a scikit-learn transformer, so whatever it
learns is learned from the training fold only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline


class Residualize(BaseEstimator, TransformerMixin):
    """Regress the last `n_cov` columns (covariates such as age and sex) out of every gene.

    Coefficients come from the training fold; held-out samples are adjusted
    with those same coefficients. The covariate columns are then dropped, so
    the forest can only see expression the covariates don't explain.
    """

    def __init__(self, n_cov=0):
        self.n_cov = n_cov

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        if self.n_cov == 0:
            return X, np.ones((len(X), 1))
        return X[:, :-self.n_cov], np.column_stack([np.ones(len(X)), X[:, -self.n_cov:]])

    def fit(self, X, y=None):
        G, C = self._split(X)
        self.cmean_ = C.mean(0)
        self.beta_, *_ = np.linalg.lstsq(C, G, rcond=None)
        return self

    def transform(self, X):
        G, C = self._split(X)
        if self.n_cov:
            C = np.where(np.isnan(C), self.cmean_, C)
        return G - C[:, 1:] @ self.beta_[1:]


def notebook_pipeline(n_cov=0, k=500, n_trees=300):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if n_cov:
        steps.append(("resid", Residualize(n_cov)))
    steps += [
        ("varfilt", VarianceThreshold(0.0)),
        ("select", SelectKBest(f_classif, k=k)),
        ("rf", RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                                      random_state=42, n_jobs=-1)),
    ]
    return Pipeline(steps)


GRID = {
    "select__k": [50, 200, 500],
    "rf__max_features": ["sqrt", 0.2],
    "rf__min_samples_leaf": [1, 3],
}


def _gene_importance(model, n_genes):
    """Map the fitted forest's importances back onto the original gene columns."""
    imp = np.zeros(n_genes)
    keep = np.flatnonzero(model.named_steps["varfilt"].get_support())
    keep = keep[model.named_steps["select"].get_support()]
    imp[keep] = model.named_steps["rf"].feature_importances_
    return imp


def nested_auc(X, y, n_cov=0, repeats=5, folds=5, seed=0, grid=GRID):
    """Held-out AUC from nested CV: the grid search never sees the outer test fold.

    Returns per-fold AUCs, out-of-fold probabilities averaged over repeats, and
    for every gene the fraction of outer folds whose tuned model kept it and
    its mean forest importance across those folds.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    n_genes = X.shape[1] - n_cov
    outer = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    aucs, prob_sum, prob_n = [], np.zeros(len(y)), np.zeros(len(y))
    imps = []
    for tr, te in outer.split(X, y):
        inner_folds = min(3, np.bincount(y[tr]).min())
        gs = GridSearchCV(notebook_pipeline(n_cov), grid, scoring="roc_auc",
                          cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        p = gs.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        prob_sum[te] += p
        prob_n[te] += 1
        imps.append(_gene_importance(gs.best_estimator_, n_genes))
    imps = np.array(imps)
    return dict(aucs=np.array(aucs), prob=prob_sum / np.maximum(prob_n, 1),
                kept=(imps > 0).mean(0), importance=imps.mean(0))
