
import pandas as pd
import numpy as np
from scipy.stats import t
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statistics
import sklearn
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def _get_weights_ffd(d: float, thresh: float = 1e-5) -> np.array:
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d - k + 1)
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fixed_width_frac_diff(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.DataFrame:
    w, df = _get_weights_ffd(d, thresh), {}
    width = len(w) - 1
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]): continue # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def min_frac_diff_d_val_plot(df0: pd.DataFrame, d_range: list, thresh: float = 1e-5) -> pd.DataFrame:
    out = pd.DataFrame(columns=["adfstat", "pval", "lags", "n_obs", "95%_conf", "corr"])
    for d in np.linspace(d_range[0], d_range[1], d_range[2]):
        print("d_value is {}".format(d))
        df1 = np.log(df0[["close"]]).resample('1D').last() # downcasting to daily values
        df2 = fixed_width_frac_diff(df1, d, thresh)
        corr = np.corrcoef(df1.loc[df2.index, "close"], df2["close"])[0, 1]
        df2 = adfuller(df2["close"], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]["5%"]] + [corr] # i.e. with critical values
    out.to_csv("./ffd_es1_test.csv")
    out[["adfstat", "corr"]].plot(secondary_y='adfstat')
    plt.axhline(out["95%_conf"].mean(), linewidth = 1, color = 'r', linestyle='dotted')
    plt.show()


def min_frac_diff_d_val(df0: pd.DataFrame, d_range: list, thresh: float = 1e-5, t_stat: float = -2.8623) -> pd.DataFrame:
    out = pd.DataFrame(columns=["adfstat", "pval", "lags", "n_obs", "95%_conf", "corr"])
    for d in np.linspace(d_range[0], d_range[1], d_range[2]):
        print("d_value is {}".format(d))
        df1 = np.log(df0[["close"]]).resample('1D').last() # downcasting to daily values
        df2 = fixed_width_frac_diff(df1, d, thresh)
        corr = np.corrcoef(df1.loc[df2.index, "close"], df2["close"])[0, 1]
        df2 = adfuller(df2["close"], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]["5%"]] + [corr] # i.e. with critical values

    d_val = out["adfstat"].loc[out["adfstat"] < t_stat].idxmax()

    return d_val


def gen_meta_labels(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Implementing  meta-labeling scheme used in Marcos-Lopez De Pardo book
    events.index: trade start time
    events["t1"]: trade end time
    events["side"]: trade side (long/short)
    """

    # Aligning prices with events
    events_ = events.dropna(subset=["trade_entry"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # creating out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values/px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"]*=events_["side"]
    # Adding meta-labeling
    out["bin"] = np.sign(out["ret"])

    # labeling/masking unprofitable events as zero
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0

    return out


def gen_cust_meta_labels(df: pd.Series, groupby_key: str, label_thresh: float) -> pd.DataFrame:
    """Implementing custom meta-labeling scheme, not the one used in
        Marcos-Lopez De Pardo book
        Treat trade start datetime and end datetime as samples, and assign,
        meta labels (slippage cost adjusted), based on actual outcomes. Then
        train a custom classifier to identify size of the trade [0, 1]

    """
    if groupby_key == "daily":
        df["date"] = df["datetime"].dt.date
        df_agg = df.groupby(["date"]).agg({
            "strat_ret": "sum",
            "close": "last",
            "strat_trades": "sum"
        })
        df_agg["labels"] = [1 if x >= label_thresh else 0 for x in df_agg["strat_ret"].tolist()]

        #todo: add multi-class meta-labels, binary labels are extremely sub-optimal

        return df_agg

    elif groupby_key == "trades":
        #todo: here find start and end points of a particular trades, groupby for unique trades
        # and generate meta-labels for each unique trade

        pass


def drop_labels(events: pd.DataFrame, min_pct_thresh: float = 0.5) -> pd.DataFrame:
    """events contains the labels associated with relevant timestamps"""
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min()>min_pct_thresh or df0.shape[0]<3:
            break
        print("dropped labels are {}_{}".format(df0.argmin(), df0.min()))
        events = events[events["bin"]!=df0.argmin()]
    return events


def cumsum_filter_sampling(close: pd.Series, filter_size: float = 0.02) -> pd.Series:
    """
    Note the filter size must be parameterized, i.e. it should be a function
    of call and put IVs.
    :returns a set of events for the point of interest, we will only trade
    on these events, based on feature values available at this point of time,
    including price, volume, IV, alternate data features. Apply a layer of meta-label model
    to eliminate false positives
    """

    t_events, s_pos, s_neg = [], 0, 0
    diff = close.pct_change(1)
    t_neg_events, t_pos_events = [], []
    for i in diff.index[1:]:
        # tmp = diff.loc[i]
        s_pos, s_neg = max(0, s_pos + diff.loc[i]), min(0, s_neg + diff.loc[i])
        if s_neg < - filter_size:
            s_neg = 0; t_events.append(i)
            t_neg_events.append(i)  # downward pivot points (can use these as swing points)
        elif s_pos > filter_size:
            s_pos = 0; t_events.append(i)
            t_pos_events.append(i)  # upward pivot points (can use these as swing points)
    return close, pd.DatetimeIndex(t_events), pd.DatetimeIndex(t_neg_events), pd.DatetimeIndex(t_pos_events)


def scipy_class_weights(class_weight: str, classes: np.array) -> np.array:
    """pg 72: Estimate class weights for unbalanced datasets.
       https://towardsdatascience.com/practical-tips-for-class-imbalance-in-binary-classification-6ee29bcdb8a7
       https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
       checkout: https://github.com/scikit-learn/scikit-learn/issues/4324
    """
    return compute_class_weight(class_weight, classes)


def student_t_stdev(close: pd.Series, rolling_window: int) -> pd.Series:
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    https://www.investopedia.com/terms/t/tdistribution.asp
    https://www.statlect.com/probability-distributions/student-t-distribution
    https://stats.stackexchange.com/questions/52609/fitting-t-distribtution-to-financial-data
    https://stackoverflow.com/questions/39000357/fit-t-distribution-using-scipy-with-predetermined-mean-and-stdloc-scale
    https://stackoverflow.com/questions/52207964/fitting-data-with-a-custom-distribution-using-scipy-stats
    check the link above

    :returns t-distribution mean and variance
    """
    t_stat = t.fit(close)
    return t_stat[1], t_stat[2]


def time_decay(df: pd.Series, clf_last: int = 1) -> pd.DataFrame:
    """needs implementation"""
    pass


def abs_return_attribution(t1: pd.DatetimeIndex, num_co_events: pd.Series,
                           close: pd.Series, molecule: pd.Series) -> pd.DataFrame:
    """needs implementation"""
    pass


if __name__ == '__main__':

    # es1_filepath = './ES1.csv'
    # df0 = pd.read_csv(es1_filepath, index_col=0, parse_dates=True)
    # d_val = min_frac_diff_d_val(df0=df0, d_range=[0, 1, 21], thresh=0.01)
    # print("optimal value of d* is {}".format(d_val))
    # todo: test the custom meta-label function, use the sample generated file
    filepath, groupby_key, thresh = "bband_bo.csv", "daily", 0.0006
    sam_df = pd.read_csv(filepath).drop(columns=["Unnamed: 0"])
    sam_df["datetime"] = pd.to_datetime(sam_df["datetime"])
    df = gen_cust_meta_labels(sam_df, groupby_key, thresh)



