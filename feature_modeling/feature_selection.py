import pandas as pd
import numpy as np
import scipy
import allel
import scipy.stats as ss
from sklearn.model_selection._split import _BaseKFold
import shap
import xgboost
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error
from feature_modeling.onc import get_onc_clusters


class PurgedKfold(_BaseKFold):
    def __init__(self, n_splits=3, t1=None, pct_embargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be pandas series')
        super(PurgedKfold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg:]))
            yield train_indices, test_indices


def num_of_bins(nObs, corr=None):
    if corr is None:  # univariate case
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2) ** .5) ** (1 / 3.)
        b = round(z / 6. + 2. / (3 * z) + 1. / 3)
    else:  # bivariate case
        b = round(2 ** -.5 * (1 + (1 + 24 * nObs / (1. - corr ** 2)) ** .5) ** .5)
    return int(b)


def var_of_info_dist(x, y, norm=False):
    if np.array_equal(x, y):
        return 0

    bXY = num_of_bins(x.shape[0])
    cXY = np.histogram2d(x, y, bXY)[0]

    iXY = mutual_info_score(None, None, contingency=cXY)  # mutual information
    hX = ss.entropy(np.histogram(x, bXY)[0])  # marginal
    hY = ss.entropy(np.histogram(y, bXY)[0])  # marginal
    vXY = hX + hY - 2 * iXY  # variation of information
    if norm:
        hXY = hX + hY - iXY  # joint
        vXY /= hXY  # normalized variation of information
    if vXY < 0:
        vXY = 0
    return vXY


def var_of_info_dist_matrix(X, norm=False):
    "gen pairwise results "
    dist = allel.pairwise_distance(X.values, var_of_info_dist)
    updated_dist = scipy.spatial.distance.squareform(
        dist)  # this can only be since since VI is a metric, see presentation
    return updated_dist


def onc_clustering_labels(X, dist_type="non_linear"):
    """
    # https: // clusteringjl.readthedocs.io / en / latest / hclust.html
    # https: // scikit - learn.org / stable / modules / generated / sklearn.cluster.AgglomerativeClustering.html
    # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.cluster.hierarchy.linkage.html
    # setting distance_threshold=0 ensures we compute the full tree.
    """

    if dist_type == "linear":
        X_ = X.T
        correl_mat = np.corrcoef(X_.values)
        dist = ((1 - correl_mat) * 0.5) ** 0.5
    else:
        dist = var_of_info_dist_matrix(X, norm=True)

    num_clusters = len(ONC_algorithm(X)[1])
    model = AgglomerativeClustering(affinity="precomputed", linkage='average',
                                    n_clusters=num_clusters)  # this finds optimal number of clusters
    model = model.fit_predict(dist)
    return pd.Series(model, index=X.columns).to_frame("clusters")


def ONC_algorithm(X, type="linear", repeat=1):
    """
    # ONC uses kmeans clustering, not hierarchial one, so mlfinlab code base
    is altered to use agglemorative clustering
    # https://mlfinlab.readthedocs.io/en/latest/implementations/clustering.html
    # for now works well with correlations, not so much with VI
    :param type:
    :return:
    """

    if type == "linear":
        X = X.T
        correl_mat = np.corrcoef(X.values)
        dist = correl_mat
    else:
        dist = var_of_info_dist_matrix(X, norm=True)
    dist = pd.DataFrame(dist)
    clust_obj = get_onc_clusters(dist, repeat=repeat)

    return clust_obj


def feat_imp_mda_cluster(clf, X, y, cv, sample_weight, t1, pct_embargo, scoring='neg_log_loss'):
    if scoring not in ['neg_log_loss', 'accuracy', 'mean_absolute_error']:
        raise Exception("wrong scoring method")
    cv_gen = PurgedKfold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        elif scoring == 'mean_absolute_error':
            pred = fit.predict(X1)
            scr0.loc[i] = mean_absolute_error(y1, pred, sample_weight=w1.values)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        cluster_names = onc_clustering_labels(X0, dist_type="non_linear").reset_index()
        cluster_df = cluster_names.groupby(["clusters"])
        for key, val in cluster_df:
            d_f = val
            X1_ = X1.copy(deep=True)
            for rows in d_f["index"].iteritems():
                np.random.shuffle(X1_[rows[1]].values)
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                for rows in d_f["index"].iteritems():
                    scr1.loc[i, rows[1]] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            if scoring == 'mean_absolute_error':
                pred = fit.predict(X1_)
                for rows in d_f["index"].iteritems():
                    scr1.loc[i, rows[1]] = mean_absolute_error(y1, pred, sample_weight=w1.values)
            else:
                pred = fit.predict(X1_)
                for rows in d_f["index"].iteritems():
                    scr1.loc[i, rows[1]] = accuracy_score(y1, pred, sample_weight=w1.values)
        imp = (-scr1).add(scr0, axis=0)
        if scoring == 'neg_log_loss':
            imp = imp / -scr1
        elif scoring == 'mean_absolute_error':
            imp = imp / -scr1
        else:
            imp = imp / (1. - scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** (-0.5)}, axis=1)
    imp = imp.sort_values(by=["mean"], ascending=False)
    return imp, scr0.mean()


def run_clustered_mda_feat(X_1, y_1, cross_validation, pct_embargo, scoring, n_estimators=1000, max_samples=1.,
                           min_w_leaf=0.):
    y = y_1.copy()
    X = X_1.copy()

    y["w"] = 1. / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index)

    clf = RandomForestClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced', min_weight_fraction_leaf=min_w_leaf)

    clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_features=1.,
                            max_samples=max_samples, oob_score=True, n_jobs=1)

    imp, oos = feat_imp_mda_cluster(clf, X, y["labels"], cv=cross_validation, sample_weight=y["w"],
                                    t1=y['t1'], pct_embargo=pct_embargo, scoring=scoring)

    return imp, oos


def run_clustered_mda_feat_regressor(X_1, y_1, cross_validation, pct_embargo, scoring, n_estimators=1000,
                                     max_samples=1., min_w_leaf=0., labels='labels'):
    y = y_1.copy()
    X = X_1.copy()
    y["w"] = 1. / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index)
    clf = RandomForestRegressor(random_state=0)

    clf = BaggingRegressor(base_estimator=clf, n_estimators=n_estimators, max_features=1.,
                           max_samples=max_samples, oob_score=True, n_jobs=1)
    imp, oos = feat_imp_mda_cluster(clf, X, y[labels], cv=cross_validation, sample_weight=y["w"],
                                    t1=y['t1'], pct_embargo=pct_embargo, scoring=scoring)
    return imp, oos


def feat_imp_shapley(clf, X_full, y_full, cv, t1, pct_embargo, scoring='neg_log_loss'):
    cv_gen = PurgedKfold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    shap_feat_imp = []
    for i, (train, test) in enumerate(cv_gen.split(X=X_full)):
        # print("in CV Shapley")
        X, y = X_full.iloc[train, :], y_full.iloc[train]
        model = clf.fit(X, y)
        explainer = shap.TreeExplainer(model)
        features = X
        features.index = list(range(features.shape[0]))
        shap_values = explainer.shap_values(features)
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X.columns, vals)),
                                          columns=['features', 'feature_importance_vals'])
        feature_importance = feature_importance.set_index("features")
        shap_feat_imp.append(feature_importance)
        # feature_importance = feature_importance.sort_values(by=['feature_importance_vals'], ascending=False)
    shap_feat_imp_df = pd.concat(shap_feat_imp, axis=1)
    shap_feat_imp_mean = shap_feat_imp_df.mean(axis=1)
    shap_feat_imp_mean_df = shap_feat_imp_mean.to_frame("feature_importance")
    return shap_feat_imp_mean_df.sort_values("feature_importance", ascending=False)


def run_shap_feat(X_1, y_1, cross_validation, pct_embargo):
    y = y_1.copy()
    X = X_1.copy()
    y["w"] = 1. / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index)
    clf = xgboost.XGBClassifier()

    feat_imp = feat_imp_shapley(clf, X, y["labels"], cv=cross_validation, t1=y['t1'], pct_embargo=pct_embargo,
                                scoring='neg_log_loss')

    return feat_imp


if __name__ == '__main__':
    df = pd.read_csv("/Users/sujitkhanna/Desktop/Talos/venv/research/ml/random_forest_input.csv")
    X = df[df.columns.tolist()[:-3]]
    Y_ART = df[['AVERAGE RETURN PER TRADE(BPS).TEST']]
    imp, oos = run_clustered_mda_feat_regressor(X, Y_ART, cross_validation=2, pct_embargo=0.,
                                                scoring="mean_absolute_error",
                                                n_estimators=100, max_samples=1., min_w_leaf=0.,
                                                labels='AVERAGE RETURN PER TRADE(BPS).TEST')


