"""
Create heuristic features, which will serve as inputs for the feature_creation.py
including intraday and daily features

https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/t-score-vs-z-score/

price-volume features:
Volume:
    1. Opening Gaps, 2. Overnight Volumes, 3. volume breakouts (daily/overnight)
Price:
    1. HL range, 2. OCR, 3. OC^2/R, 4. RVs (overall, -ve, +ve) BO, 5. IVs BO, 6. IV-RV  7. Contago/backwardation
    8. implied v realized correlations, 9. Beta BO
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import t
from sklearn.preprocessing import (power_transform, QuantileTransformer)
from feature_modeling.price_features import price_features



class RawPriceVolumeFeatures(price_features):

    def __init__(self, ohlcv_df: pd.DataFrame, lbk: int, time_frame: str, distribution: str = "normal", benchmark_df: pd.DataFrame = None,
                 implied_vol_df: pd.DataFrame = None, futures_df: pd.DataFrame = None, imp_corr_df: pd.DataFrame = None):

        self.ohlcv_df = ohlcv_df
        self.lbk = lbk
        self.benchmark_df = benchmark_df
        self.implied_vol_df = implied_vol_df
        self.futures_df = futures_df
        self.imp_corr_df = imp_corr_df
        self.distribution = distribution
        self.time_frame = time_frame
        self.pv_obj = price_features(ohlcv_df, self.time_frame)

    def _on_volumes(self, open_time: str, close_time: str):
        on_df = self.ohlcv_df
        on_df["date"] = self.ohlcv_df.index.strftime('%Y-%m-%d')
        prev_day = on_df.between_time(close_time, '23:59:00').groupby(["date"])["volume"].sum().fillna(0).to_frame()

        cur_day = on_df.between_time('00:00:00', open_time).groupby(["date"])["volume"].sum().fillna(0).to_frame()

        total_sum = cur_day.join(prev_day, how="left", lsuffix='_cur', rsuffix='_prev').fillna(0)
        on_volumes = pd.DataFrame({"volume":total_sum["volume_cur"].iloc[1:].values +
                                            total_sum["volume_prev"].iloc[:-1].values},
                                  index = total_sum["volume_cur"].iloc[1:].index)


        return on_volumes.rename(columns={"volume":"on_volume"})

    def on_volumes(self, open_time: str, close_time: str):
        on_vol = self._on_volumes(open_time, close_time)
        return on_vol.rolling(self.lbk).sum()

    def _daily_volumes(self, open_time: str, close_time: str):
        "needs adjustment similar to on volumes"
        on_vol = self._on_volumes(open_time, close_time)
        vol_df = self.ohlcv_df
        vol_df["date"] = self.ohlcv_df.index.strftime('%Y-%m-%d')
        cur_day_vol = vol_df.between_time(open_time, close_time).groupby(["date"])["volume"].sum().fillna(0).to_frame()
        daily_vol = on_vol.rename(columns={"on_volume":"volume"}) + cur_day_vol
        # return daily_vol
        return daily_vol.ffill()

    def daily_volumes(self, open_time: str, close_time: str):
        daily_vol = self._daily_volumes(open_time, close_time)
        return daily_vol.rolling(self.lbk).sum()

    def gen_gaps(self):
        self.pv_obj.create_gaps()
        return self.pv_obj.gaps.set_index(["date"])

    def gen_hl_range(self):
        """only for the day session traded times"""
        self.pv_obj.create_daily_hl_range()
        return self.pv_obj.daily_hl_range["hl_range_norm"]

    def _daily_agg(self, obj):
        obj.create_daily_ohlcv()
        daily_df = obj.daily_ohlcv
        daily_df["date"] = daily_df["datetime"].dt.date
        return daily_df.groupby(["date"]).agg({
            'open':'first',
            'high': 'first',
            'low': 'first',
            'close': 'first',
        })

    def gen_ocr_oc2_r(self):
        """only for the day session traded times"""
        ocr_df = self._daily_agg(self.pv_obj)
        return ((ocr_df["close"] - ocr_df["open"])
                / (ocr_df["high"] - ocr_df["low"])).to_frame("ocr"),\
               (((ocr_df["close"] - ocr_df["open"])**2)
                / (ocr_df["high"] - ocr_df["low"])).to_frame("oc2r")

    def gen_daily_rv(self):
        """include RVs that consider ohlc bars as well"""
        daily_df = self._daily_agg(self.pv_obj)
        daily_ret = daily_df["close"].pct_change(1)
        if self.distribution == "normal":
            return daily_ret.rolling(self.lbk).std()
        elif self.distribution == "student_t":
            def gen_roll_t_distrib(df):
                return t.fit(df)[2]
            return daily_ret.rolling(self.lbk).apply(gen_roll_t_distrib)

    def gen_iv(self, sample_freq):
        """return sampled dataframe"""
        return self.implied_vol_df.resample(sample_freq).agg({
            "close":"last"
        })

    def gen_cont_back(self):
        """futures data is assumed to have daily frequency"""
        return self.ohlcv_df["second_month"] / self.ohlcv_df["first_month"]

    def gen_beta(self):
        """compute beta on daily returns"""
        bench_obj = price_features(self.benchmark_df, self.time_frame)
        daily_df, daily_bench_df = self._daily_agg(self.pv_obj), self._daily_agg(bench_obj)
        cov_df = daily_df.join(daily_bench_df, how="left", lsuffix='_asset',
                               rsuffix='_benchmark').ffill()[["close_asset", "close_benchmark"]].pct_change(1)
        return (cov_df.rolling(self.lbk).cov().unstack()["close_asset"]["close_benchmark"]
                / cov_df["close_benchmark"].rolling(self.lbk).var())

    def gen_realized_corr(self):
        pass

    def gen_ibs(self):
        """only for the day session traded times"""
        ibs_df = self._daily_agg(self.pv_obj)
        return ((ibs_df["close"] - ibs_df["low"])
                / (ibs_df["high"] - ibs_df["low"])).to_frame("ibs")


class ProcessedPriceVolumeFeatures(RawPriceVolumeFeatures):
    def __init__(self, ohlcv_df: pd.DataFrame, lbk_1: int, lbk_2: int, time_frame: str, roll_lbk: int, feature_distribution:str = 'normal', score_distribution: str = None,
                 pctile_distribution: str = None, benchmark_df: pd.DataFrame = None, implied_vol_df: pd.DataFrame = None,
                 futures_df: pd.DataFrame = None, imp_corr_df: pd.DataFrame = None, n_quantiles = 10):

        self.feature_obj_lbk1 = RawPriceVolumeFeatures(ohlcv_df, lbk_1, time_frame, feature_distribution, benchmark_df,
                                                       implied_vol_df, futures_df, imp_corr_df)
        self.feature_obj_lbk2 = RawPriceVolumeFeatures(ohlcv_df, lbk_2, time_frame, feature_distribution, benchmark_df,
                                                       implied_vol_df, futures_df, imp_corr_df)
        self.score_distribution = score_distribution
        self.pctile_distribution = pctile_distribution
        self.roll_lbk = roll_lbk
        self.n_quantiles = n_quantiles

    def gen_feature_ratio(self, func_name: object, *args):
        """This needs reworking"""
        func_1, func_2 = getattr(self.feature_obj_lbk1, func_name),\
                         getattr(self.feature_obj_lbk2, func_name)
        ratio = func_1(*args)/func_2(*args)
        if isinstance(ratio, pd.DataFrame):
            return ratio.rename(columns={ratio.columns[0]: f"{ratio.columns[0]}_ratio"})
        else:
            return ratio.to_frame(f"{func_name}_ratio")

    def gen_feat_score(self, df: pd.Series):
        if self.score_distribution == "normal":
            def zscore_func(x):
                return (x[-1] - x[:-1].mean()) / x[:-1].std(ddof=0)
            return df.rolling(self.roll_lbk).apply(zscore_func)
        elif self.score_distribution == "t_dist":
            def t_stat(df0):
                return (df0[-1].values - df0.values.mean())/(df0.std()/math.sqrt(df0.shape[0]))
            return df.rolling(self.roll_lbk).apply(t_stat)
        elif self.score_distribution == "other":
            print("think of other fat-tailed distributions")

    def gen_percentile_scores(self, df: pd.Series):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform
        return df.rolling(self.roll_lbk).apply(lambda x: pd.Series(x).rank(method='average', pct=True).values[-1])
        """

        if self.pctile_distribution == "uniform":
            def uniform_quantile(df0):
                transformer = QuantileTransformer(n_quantiles=self.n_quantiles)
                return transformer.fit_transform(np.asarray(df0).reshape(-1, 1))[-1]
            return df.rolling(self.roll_lbk).apply(uniform_quantile)

        elif self.pctile_distribution == "gaussian":
            def gaussian_quantile(df0):
                transformer = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution='normal')
                return transformer.fit_transform(np.asarray(df0).reshape(-1, 1))[-1]
            return df.rolling(self.roll_lbk).apply(gaussian_quantile)

        elif self.pctile_distribution == "yeojo":
            def yeojo_quantile(df0):
                yj_ary = power_transform(np.asarray(df0).reshape(-1, 1), method='yeo-johnson')[-1]
                return yj_ary
            return df.rolling(self.roll_lbk).apply(yeojo_quantile)

        elif self.pctile_distribution == "other":
            return None
