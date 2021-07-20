'''
Functions below are extremely inefficient, refactor this file
to created optimized and high performing functions,
1. vectorize operations whereever possible
2. eliminate unnecessary function calls
3. use additional libraries, that will optimize the methods
'''

import pandas as pd
import numpy as np
import datetime as dt


MASTER_START = dt.datetime.strptime('09:15:00', '%H:%M:%S').time()
MASTER_END = dt.datetime.strptime('16:00:00', '%H:%M:%S').time()

class price_features:
    def __init__(self, raw_df: pd.DataFrame, bars: str) -> None:
        self.raw_df = raw_df
        self.bars = bars
        self.price_ret = None
        self.endpoints = None
        self.gaps = None
        self.ohlcv = None
        self.daily_ohlcv = None
        self.prevday_ohlcv = None
        self.daily_hl_range = None
        self.daily_oc_range=None
        self.gen_ohlcv_bars()
        self.gen_returns()

    def gen_ohlcv_bars(self) -> None:
        '''
        generates ohlcv based on the frequency of bars
        from the input parameter,
        assigns it to class variable
        todo: check if you want to return a df or array
        '''
        ohlcv_df = self.raw_df.resample(self.bars).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        # self.ohlcv = ohlcv_df[ohlcv_df['datetime'].dt.time.between(MASTER_START, MASTER_END)]
        self.ohlcv = ohlcv_df

        self.ohlcv = self.ohlcv[~self.ohlcv['close'].isna()]
        self.ohlcv.index = list(range(self.ohlcv.shape[0]))
        #decide if you want to return a Dataframe or an Numpy Array


    def gen_returns(self) -> None:
        '''
        generates percentage returns from class ohlcv df
        '''
        self.price_ret = self.ohlcv['close'].pct_change(1)

        '''
        return_ary = np.asarray(price_ret).astype(float)
        return_ary[0] = 0
        return_ary_new = np.reshape(return_ary, (len(return_ary, )))
        return return_ary_new
        '''


    def gen_endpoints(self) -> None:
        '''
        generates end points in the day used for closing down
        intraday strategy positions at the EOD
        '''

        datetime_df = pd.DataFrame({'datetime':self.ohlcv['datetime'],
                                    'endpoints':[i for i in range(len(self.ohlcv['datetime']))]})
        datetime_df['date'] = datetime_df['datetime'].dt.date

        self.endpoints = datetime_df.groupby(['date']).agg({
            'endpoints': 'last'
        }).reset_index()


    def create_gaps(self) -> None:
        '''
        Generates daily gap features, used extensively in
        price action strategies
        '''

        df = self.ohlcv
        df['date'] = df['datetime'].dt.date
        df_daily = df.groupby(['date']).agg({
            'open':'first',
            'close':'last'
        }).reset_index()

        self.gaps = pd.DataFrame({'date':df_daily['date'][1:],'gaps':(df_daily['open'][1:].values - df_daily['close'][:-1].values)/df_daily['close'][:-1].values})


    def create_daily_ohlcv(self) -> None:
        '''
        creates daily ohlcv data primarily used for
        creating range based features
        '''

        df = self.ohlcv
        df['date'] = df['datetime'].dt.date
        ohlcv_day = df.groupby(['date']).agg({
            'open':'first',
            'high':'max',
            'low': 'min',
            'close':'last',
            'volume':'sum'
        }).reset_index()

        tmp_lagger = ohlcv_day.shift(1)
        lagged_ohlcv = pd.DataFrame({'date': ohlcv_day['date'],
                                     'open':tmp_lagger['open'].iloc[1:], 'high':tmp_lagger['high'].iloc[1:],
                                     'low':tmp_lagger['low'].iloc[1:], 'close':tmp_lagger['close'].iloc[1:],
                                     'volume':tmp_lagger['volume'].iloc[1:]})

        #make change to
        self.daily_ohlcv = pd.merge(df[['datetime','date']], ohlcv_day, how='left', on=['date'])
        self.prevday_ohlcv = pd.merge(df[['datetime','date']], lagged_ohlcv, how='left', on=['date'])



    def create_n_day_high_low(self, lbk) -> pd.DataFrame:
        '''
        creating n day high low time series, which will
        be used later to compute n day ranges
        todo: see if n day ranges are to be calculated in this class
        '''

        df = self.ohlcv
        df['date'] = df['datetime'].dt.date
        df_daily = df.groupby(['date']).agg({
            'low':'min',
            'high':'max'
        }).reset_index()

        return pd.DataFrame({'date':df_daily['date'],
                             'low':df_daily['low'].rolling(lbk).min(),
                             'high':df_daily['high'].rolling(lbk).max()})


    def create_daily_hl_range(self):
        '''
        creating daily hl range series both raw and normalized
        will be used as filters and for developing allocation models
        '''

        df = self.ohlcv
        df['date'] = df['datetime'].dt.date
        df_daily = df.groupby(['date']).agg({
            'low':'min',
            'high':'max'
        }).reset_index()
        df_daily['hl_range'], df_daily['hl_range_norm'] = (df_daily['high'] - df_daily['low']), \
                                                          ((df_daily['high'] - df_daily['low'])*2)/(((df_daily['high'] + df_daily['low'])))

        self.daily_hl_range = pd.DataFrame({'hl_range':df_daily['hl_range'].values,
                                           'hl_range_norm':df_daily['hl_range_norm'].values}, index=df_daily['date'].values)


    def create_daily_oc_range(self, lbk):
        '''
        creating daily open to close range series both raw and normalized
        will be used as filters and for developing allocation models
        '''

        df = self.ohlcv
        df['date'] = df['datetime'].dt.date
        df_daily = df.groupby(['date']).agg({
            'open':'first',
            'close':'last'
        }).reset_index()
        df_daily['oc_range'], df_daily['oc_range_norm'] = abs(df_daily['open'] - df_daily['close']), \
                                                          abs((df_daily['open'] - df_daily['close'])/(2*((df_daily['open'] + df_daily['close']))))

        self.daily_oc_range = pd.DataFrame({'date':df_daily['date'],
                                           'oc_range':df_daily['oc_range'],
                                           'oc_range_norm':df_daily['oc_range_norm']})

    def create_daily_ibs_ind(self) -> pd.DataFrame:
        '''
        calculate daily ibs based on the formula below,
        IBS  =  (Close – Low) / (High – Low),for detailed description check
        -> http://jonathankinlay.com/2019/07/the-internal-bar-strength-indicator/
        '''

        self.create_daily_ohlcv()
        daily_ibs = pd.DataFrame({'date':self.daily_ohlcv['date'], 'daily_ibs':(
            (self.daily_ohlcv['close'] - self.daily_ohlcv['low'])/(self.daily_ohlcv['high'] - self.daily_ohlcv['low'])
        )})
        return daily_ibs

    def n_day_breakout_distance(self)->pd.DataFrame:
        '''
        calculate the distance of current price to it's last n-day
        high/low prices:-> to be used as a part of component strategy
        '''
        pass

    def create_n_day_ibs_ind(self, lbk)->pd.DataFrame:
        '''
        calculating the n day ibs indicator using the same mechanics
        as daily ibs, useful for stock selection, i.e. reversionary or
        momentum stocks
        '''

        self.create_daily_ohlcv()
        last_close = self.daily_ohlcv['close'].rolling(lbk).apply(lambda x:x[-1])
        min_low = self.daily_ohlcv['low'].rolling(lbk).min()
        max_high = self.daily_ohlcv['high'].rolling(lbk).min()


        return pd.DataFrame({'date':self.daily_ohlcv['date'],
                             'n_day_ibs':(last_close-min_low)/(max_high-min_low)})


    def trailing_sl(self, trade_sig, tsl) -> np.array:
        '''
        Trailing stop loss function, developed a couple of years ago,
        this needs more testing on logic accuracy and integration
        '''

        sig_trades, strat_price = trade_sig, np.array(self.ohlcv['close']).astype(float)
        long_pt, short_pt = np.asarray(np.zeros(sig_trades.shape).astype(object)), np.asarray \
            (np.zeros(sig_trades.shape).astype(int))
        long_high, short_low = np.asarray(np.zeros(sig_trades.shape).astype(float)), np.asarray \
            (np.zeros(sig_trades.shape).astype(float))

        "calculate the trade high and low at every point in time for a trade"
        for i in range(0, len(sig_trades)):
            if sig_trades[i] == 1 and sig_trades[i - 1] == 0:
                long_pt[i] = i
            if sig_trades[i] == -1 and sig_trades[i - 1] == 0:
                short_pt[i] = i

        for i in range(0, len(sig_trades)):
            if sig_trades[i] == 1 and sig_trades[i - 1] == 1:
                long_pt[i] = long_pt[i - 1]
                tmp = long_pt[i]
                long_high[i] = max(strat_price[int(long_pt[i]):i])
            elif sig_trades[i] == 1 and sig_trades[i - 1] == 0:
                long_high[i] = strat_price[i]
            if sig_trades[i] == -1 and sig_trades[i - 1] == -1:
                short_pt[i] = short_pt[i - 1]
                short_low[i] = min(strat_price[int(short_pt[i]):i])
            elif sig_trades[i] == -1 and sig_trades[i - 1] == 0:
                short_low[i] = strat_price[i]

        new_sig = np.copy(sig_trades)

        for i in range(0, len(sig_trades) - 1):
            if sig_trades[i] == 1 and ((long_high[i] / strat_price[i]) - 1) >= tsl:
                new_sig[i] = 0
            elif sig_trades[i] == 1 and sig_trades[i + 1] != 0 and sig_trades[i - 1] != 0 and new_sig[i - 1] == 0:
                new_sig[i] = 0
            if sig_trades[i] == -1 and ((short_low[i] / strat_price[i]) - 1) <= (-1) * tsl:
                new_sig[i] = 0
            elif sig_trades[i] == -1 and sig_trades[i + 1] != 0 and sig_trades[i - 1] != 0 and new_sig[i - 1] == 0:
                new_sig[i] = 0

        return new_sig

