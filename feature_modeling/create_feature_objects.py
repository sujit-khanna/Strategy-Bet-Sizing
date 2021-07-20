import attr
from feature_modeling import feature_heuristics as fh
import pandas as pd

START_TIME, END_TIME = '9:15:00', '16:00:00'

@attr.s(auto_attribs=True, frozen=True)
class InputParameters:
    table_name: str
    ticker: str
    bench_table_name: str
    bench_ticker: str


@attr.s(auto_attribs=True, frozen=True)
class FeatureParameters:
    ohlcv_df: pd.DataFrame
    distribution: str
    benchmark_df: pd.DataFrame
    lbk_long: int
    lbk_short: int
    time_frame: str
    roll_lbk: int
    feature_distribution: str
    score_distribution: str
    pctile_distribution: str


def get_feat_objects(feat_obj: FeatureParameters) -> object:

    raw_obj = fh.RawPriceVolumeFeatures(feat_obj.ohlcv_df, feat_obj.lbk_long,
                                        feat_obj.time_frame, feat_obj.distribution,
                                        feat_obj.benchmark_df)

    return raw_obj


def get_processed_feat(feat_obj: FeatureParameters) -> object:
    raw_obj = fh.ProcessedPriceVolumeFeatures(feat_obj.ohlcv_df, feat_obj.lbk_long, feat_obj.lbk_short,
                                              feat_obj.time_frame, feat_obj.roll_lbk, feat_obj.feature_distribution,
                                              feat_obj.score_distribution, feat_obj.pctile_distribution,
                                              feat_obj.benchmark_df)
    return raw_obj


def process_raw_files(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index('datetime')

    return df


def load_obj_data(feat_dict: dict) -> FeatureParameters:
    ohlcv_df = process_raw_files(feat_dict["ticker_file_path"])
    benchmark_df = process_raw_files(feat_dict["benchmark_file_path"])
    return FeatureParameters(ohlcv_df,
                             feat_dict["distribution"],
                             benchmark_df,
                             feat_dict["lbk_long"],
                             feat_dict["lbk_short"],
                             feat_dict["time_frame"],
                             feat_dict["roll_lbk"],
                             feat_dict["feature_distribution"],
                             feat_dict["score_distribution"],
                             feat_dict["pctile_distribution"])


def read_data(input_dict: dict) -> InputParameters:
    return InputParameters(input_dict["table_name"],
                           input_dict["ticker"],
                           input_dict["bench_table_name"],
                           input_dict["bench_ticker"])


def run(feat_dict: dict) -> (object, object):
    feat_obj = load_obj_data(feat_dict)
    raw_obj = get_feat_objects(feat_obj)
    proc_obj = get_processed_feat(feat_obj)
    return raw_obj, proc_obj


if __name__ == '__main__':
    # input_dict = {"table_name":"vix_etf_intraday", "ticker":"vxx", "bench_table_name": "vix_index_intraday",
    #               "bench_ticker": "vix"}
    feat_dict = {"distribution": "normal", "lbk_long": 20,
                 "lbk_short": 5, "time_frame": "5T", "roll_lbk":20, "feature_distribution": "normal",
                 "score_distribution": "normal", "pctile_distribution": "yeojo",
                 "ticker_file_path":"data/prices/VXX_updated_2020.csv",
                 "benchmark_file_path":"data/prices/SPY_2020.csv"}

    run(feat_dict)

    # run(input_dict, feat_dict)


