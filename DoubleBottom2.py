import re
import sys
import time

import numpy as np
import pandas as pd

import pytz
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelmin, argrelmax

from multiprocessing import Pool

import talib as ta
import MetaTrader5 as mt5

# in order to call a point local min/max, how many neighboring candles will be considered?
local_extrema_window = 3
# proportion of previous candles to be upper than the first identified low point (LHL pattern is searched for at the
# end of a falling trend)
preceding_falling_trend_pct = 0.8
# period of the Average True Range indicator
ATR_period = 14
# multiplier of ATR value
ATR_multiplier = 0.5
ATR_multiplier_max = 3
# what is the max width of a pattern that we allow?
pattern_max_width = 50
# ticker list to check
ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
               'CADCHF', 'CHFPLN', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'CHFSGD', 'AUDCAD', 'AUDJPY',
               'AUDNZD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURNOK', 'EURNZD', 'EURSEK', 'GBPAUD', 'GBPCAD', 'GBPNZD',
               'NZDCAD', 'NZDJPY', 'USDNOK', 'USDSEK', 'AUDDKK', 'EURHUF', 'EURMXN', 'EURPLN', 'EURTRY', 'EURZAR',
               'GBPNOK', 'GBPPLN', 'GBPSEK', 'NOKSEK', 'PLNJPY', 'USDMXN', 'USDHUF', 'USDPLN', 'USDTRY', 'USDZAR',
               'EURRUB', 'USDRUB', 'USDILS', 'USDCNH', 'GBPZAR', 'GOLD', 'SILVER', 'WTI', 'BRENT', 'NAT.GAS', 'GOLDoz',
               'GOLDgr', 'GOLDEURO', 'SILVEREURO', 'AUDPLN', 'AUDSGD', 'EURSGD', 'GBPSGD', 'NZDSGD', 'USDDKK', 'SGDJPY',
               'USDSGD', 'EURHKD', 'GBPDKK', 'USDHKD', '#AUS200', '#Swiss20', '#Spain35', '#ChinaA50', '#ChinaHShar',
               '#Euro50', '#France120', '#France40', '#Germany30', '#Germany50', '#GerTech30', '#Holland25',
               '#HongKong50', '#Japan225', '#UK100', '#US30', '#USNDAQ100', '#USSPX500', 'EURDKK', 'EURCZK', 'USDCZK',
               'BTCUSD', 'ETHUSD', 'LTCUSD', 'USDTHB', 'BCHUSD', 'XRPUSD', 'PLATINUM', '#UKmid250',
               '#UKOil_N21', '#Corn_N21', '#SBean_N21', '#Wheat_N21', '#Cotton_N21',
               '#Sugar_N21', '#Coffee_N21', '#NGas_N21', '#USOil_N21', '#UKOil_Q21', 'DOTUSD', 'EOSUSD', 'LNKUSD',
               'XLMUSD', 'DOGUSD', '#JP225_U21', '#DJ30_U21', '#US100_U21', '#US500_U21', '#UK100_U21', '#EUR50_U21',
               '#GER30_U21', '#FRA40_U21', '#US$idx_U21', '#SWI20_U21', 'BATUSD', 'BTGUSD', 'DSHUSD', 'ETCUSD',
               'IOTUSD', 'NEOUSD', 'XMRUSD', 'ZECUSD', '#Coffee_U21', '#Cocoa_U21', '#USOil_Q21', '#Corn_U21',
               '#SBean_Q21', '#Wheat_U21', '#Cotton_V21', '#Sugar_V21', '#UKOil_U21', '#NGas_Q21', '#USOil_U21',
               '#UKOil_V21', '#NGas_U21']

# ticker_list = ['#JP225_M21',
#                '#DJ30_M21', '#US100_M21', '#US500_M21', '#UK100_M21', '#EUR50_M21', '#GER30_M21', '#FRA40_M21',
#                '#SWI20_M21', '#NGas_M21']


# ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
#                'CADCHF', 'CHFPLN', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'CHFSGD', 'AUDCAD', 'AUDJPY',
#                'AUDNZD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURNOK', 'EURNZD', 'EURSEK', 'GBPAUD', 'GBPCAD', 'GBPNZD',
#                'NZDCAD', 'NZDJPY', 'USDNOK', 'USDSEK', 'AUDDKK', 'EURHUF', 'EURMXN', 'EURPLN', 'EURTRY', 'EURZAR',
#                'GBPNOK', 'GBPPLN', 'GBPSEK', 'NOKSEK', 'PLNJPY', 'USDMXN', 'USDHUF', 'USDPLN', 'USDTRY', 'USDZAR',
#                'EURRUB', 'USDRUB', 'USDILS', 'USDCNH', 'GBPZAR', 'AUDPLN', 'AUDSGD', 'EURSGD', 'GBPSGD', 'NZDSGD',
#                'USDDKK', 'SGDJPY', 'USDSGD', 'EURHKD', 'GBPDKK', 'USDHKD', 'EURDKK', 'EURCZK', 'USDCZK',
#                'BTCUSD', 'ETHUSD', 'LTCUSD']

ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
               'CADCHF', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'AUDCAD', 'AUDJPY', 'AUDNZD', 'CADJPY',
               'EURAUD', 'EURCAD', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDJPY', 'EURHUF', 'USDHUF',
               'EURNOK', 'USDNOK', 'EURRUB', 'USDRUB', 'BTCUSD', 'ETHUSD', 'LTCUSD']

#                '#AUS200', '#Swiss20', '#Spain35', '#ChinaA50', '#ChinaHShar',
#                '#Euro50', '#France120', '#France40', '#Germany30', '#Germany50', '#GerTech30', '#Holland25',
#                '#HongKong50', '#Japan225', '#UK100', '#US30', '#USNDAQ100', '#USSPX500',
#                ]

# ticker_list = ['BTCUSD', 'ETHUSD', 'LTCUSD']
# ticker_list = ['AUDUSD']


class DoubleTopBottom:

    def __init__(self, ticker, timeframe, local_extrema_window, preceding_falling_trend_pct, ATR_period, Aroon_period, MA_period, ATR_multiplier, pattern_max_width, timedelta_min=None):
        self.ticker = ticker
        self.timeframe = timeframe
        self.local_extrema_window = local_extrema_window
        self.preceding_falling_trend_pct = preceding_falling_trend_pct
        self.ATR_period = ATR_period
        self.Aroon_period = Aroon_period
        self.MA_period = MA_period
        self.ATR_multiplier = ATR_multiplier
        self.pattern_max_width = pattern_max_width
        self.timedelta_min = timedelta_min

        self.point = None

        self.connect_mt5()

    def connect_mt5(self):
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        self.point = mt5.symbol_info(self.ticker).point

    def get_rates(self, ticker, timeframe='M1'):
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
        if self.timedelta_min:
            utc_from = datetime.now() - timedelta(minutes=self.timedelta_min)
        else:
            utc_from = datetime.now()
        # print("From datetime: ", utc_from)
        # get 10 EURUSD M1 bars starting from 01.10.2020 in UTC time zone
        rates = mt5.copy_rates_from(ticker, eval(f"mt5.TIMEFRAME_{timeframe}"), utc_from, pattern_max_width + max(self.ATR_period, self.MA_period))
        if rates is None:
            print(f"{ticker} doesn't exist on Mt5")
            return

        df = pd.DataFrame(columns=['<DATE>_<TIME>','open','high','low','close','tickvol','volume','spread'])
        for rate in rates:
            df = df.append(pd.Series(list(rate), index=['<DATE>_<TIME>','open','high','low','close','tickvol','volume','spread']), ignore_index=True)

        df['<DATE>_<TIME>'] = pd.to_datetime(df['<DATE>_<TIME>'],unit='s')
        df['<DATE>_<TIME>'] = df['<DATE>_<TIME>'].dt.tz_localize('Etc/UTC').dt.tz_convert('Europe/Budapest').dt.tz_localize(None)

        # Calculate technical indicators : ATR, Aroon, SMA
        df[f'ATR{self.ATR_period}'] = ta.ATR(df['high'], df['low'], df['close'], self.ATR_period)

        aroon_up, aroon_down = ta.AROON(df['high'], df['low'], timeperiod=self.Aroon_period)
        df[f'Aroon{self.Aroon_period}'] = (aroon_up > aroon_down).astype(int)

        df[f'SMA{self.MA_period}'] = ta.SMA(df['close'], timeperiod=self.MA_period)

        return df

    def find_double_bottom(self, price='typical'):

        df = self.get_rates(self.ticker, self.timeframe)
        if df is None:
            return

        # create slices of df with width of 10
        df_whole = df
        window = 10
        for iii in range(0, len(df_whole)-window):
            print(iii, iii+window)
        sys.exit()
        if price not in df.columns:
            if price == 'typical':
                df[price] = (df['high'] + df['low'] + df['close']) / 3
            elif price == 'median':
                df[price] = (df['high'] + df['low']) / 2
            elif price == 'weighted':
                df[price] = (df['high'] + df['low'] + df['close']*2) / 4
            else:
                raise Exception('Invalid price argument was passed to find_double_bottom() method. Valid price arguments'
                                'are: "typical", "median", and "weighted".')

        # typical_prices = df['typical']
        df = df.set_index('<DATE>_<TIME>')

        # add an integer index column too to keep track of number of candles that make up a pattern
        df['idx'] = range(0,df.shape[0])
        df_orig = df.copy()

        # get average ATR across df
        average_ATR = np.nanmean(df[f'ATR{self.ATR_period}'])

        local_minima = df[price].iloc[argrelmin(np.array(df[price].values), order=local_extrema_window)].rename('Minima')
        local_maxima = df[price].iloc[argrelmax(np.array(df[price].values), order=local_extrema_window)].rename('Maxima')

        df = pd.concat([df, pd.Series(index=df.index, name='Extremes', dtype=object)], axis=1)

        df.loc[df.index.isin(local_minima.index), 'Extremes'] = 'L'
        df.loc[df.index.isin(local_maxima.index), 'Extremes'] = 'H'
        # df['Extremes'] = df['Extremes'].replace(np.nan, '', regex=True)

        df = df[~df['Extremes'].isnull()]
        df['Extremes-1'] = df['Extremes'].shift(1)
        df['Extremes-2'] = df['Extremes'].shift(2)

        df = df.dropna(subset=['Extremes','Extremes-1','Extremes-2',f'ATR{self.ATR_period}',f'SMA{self.MA_period}'],axis=0)
        # df = df.dropna(subset=[f'ATR{self.ATR_period}'],axis=0)

        df['Extremes'] = df['Extremes-2'] + df['Extremes-1'] + df['Extremes']
        df = df.drop(['Extremes-2', 'Extremes-1'], axis=1)


        df_orig['Extremes'] = df['Extremes']
        df_orig['Extremes'] = df_orig['Extremes'].replace(np.nan, '', regex=True)

        df = df_orig
        del df_orig

        # Condition #0: Check if there has been a recent LHL pattern
        if "LHL" not in df[df['Extremes']!='']['Extremes'].iloc[2:].tolist():
            return

        # Condition #1: Check if pattern length is less than the max. allowed ('pattern_max_width')
        datetime_last_LHL = df[df['Extremes']=='LHL'].iloc[-1,:].name

        # get idx of all three points of the pattern (L, H, L)
        df_last_LHL = df[(df.index <= datetime_last_LHL) & (df['Extremes'] != '')]
        idx_firstL = df_last_LHL.iloc[-3]['idx']
        idx_middleH = df_last_LHL.iloc[-2]['idx']
        idx_secondL = df_last_LHL.iloc[-1]['idx']


        # ensure total pattern length is less than maximally allowed ('pattern_max_width')
        if idx_secondL - idx_firstL > pattern_max_width:
            return

        # Condition #2: two Ls must be approximately on same levels
        if not np.abs(df[df['idx']==idx_firstL][price].item() - df[df['idx']==idx_secondL][price].item()) < average_ATR:
            return
        print("\tCondition #2 is met.")

        # Condition #3: middle peak (H) is at least N * ATR above the taller L where N is: ATR_multiplier
        taller_L = np.max([df[df['idx']==idx_firstL]['close'].item(), df[df['idx']==idx_secondL]['close'].item()])
        middle_H = df[df['idx']==idx_middleH]['close'].item()
        if not middle_H > taller_L + ATR_multiplier * average_ATR:
            return
        print("\t\tCondition #3 is met.")


        # Condition #4: middle peak (H) is not more than 3N * ATR above the taller L
        if not middle_H < taller_L + 3 * ATR_multiplier * average_ATR:
            return
        print("\t\t\tCondition #4 is met.")

        # Condition #5: previous N candles show in falling trend (MA > price)
        df_slice_previous_N_candles = df[(df['idx'] >= idx_firstL - 10) & (df['idx'] < idx_firstL)]
        if not all((df_slice_previous_N_candles[f'SMA{self.MA_period}'] > df_slice_previous_N_candles[price]).astype(int)):
            return
        print("\t\t\tCondition #5 is met.")

        # Condition #6: second low is higher or same level (or at most marginally lower!) than first one
        if not df[df['idx']==idx_firstL][price].item() > df[df['idx']==idx_secondL][price].item() + self.point*10:
            return
        print("\t\t\tCondition #6 is met.")

        ap = []
        for iii in [idx_firstL, idx_middleH, idx_secondL]:
            df_copy = df[(df['idx'] >= idx_middleH - pattern_max_width * 3) & (
                    df['idx'] <= idx_middleH + pattern_max_width * 3)].copy()
            df_copy[df_copy['idx'] != iii] = np.nan
            ap.append(mpf.make_addplot(df_copy[price], type='scatter', marker='.', markersize=200))

        data = df[(df['idx'] >= idx_middleH - pattern_max_width * 3)]

        mpf.plot(data, type='candle', addplot=ap,
                 title=f"{self.ticker} {self.timeframe}",
                 ylim=(data[['open', 'high', 'low', 'close']].min().min(),
                       data[['open', 'high', 'low', 'close']].max().max()))

        return

        # # connect to the trade account without specifying a password and a server
        # authorized = mt5.login(5386115, password="RqrOM2HP", server='FxPro-MT5')
        # if authorized:
        #     pass
        #     # # display trading account data 'as is'
        #     # print(mt5.account_info())
        #     # # display trading account data in the form of a list
        #     # print("Show account_info()._asdict():")
        #     # account_info_dict = mt5.account_info()._asdict()
        #     # # for prop in account_info_dict:
        #     # #     print("  {}={}".format(prop, account_info_dict[prop]))
        # else:
        #     print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
        #
        # # prepare the buy request structure
        # symbol = self.ticker
        # symbol_info = mt5.symbol_info(symbol)
        # if symbol_info is None:
        #     print(symbol, "not found, can not call order_check()")
        #     continue
        #     # mt5.shutdown()
        #     # quit()
        #
        # # if the symbol is unavailable in MarketWatch, add it
        # if not symbol_info.visible:
        #     print(symbol, "is not visible, trying to switch on")
        #     if not mt5.symbol_select(symbol, True):
        #         print("symbol_select({}}) failed, exit", symbol)
        #         # mt5.shutdown()
        #         # quit()
        #         continue
        #
        # positions = mt5.positions_get(symbol=symbol)
        # if len(positions) > 0:
        #     continue
        #
        # print(
        #     f"BUY OPPORTUNITY on {self.ticker} {self.timeframe} - LHL : {minmax_df.loc[idx, 'typical']:.{4}f}, {minmax_df.loc[idx - 1, 'typical']:.{4}f}, {minmax_df.loc[idx - 2, 'typical']:.{4}f}")
        #
        # ap = []
        # for iii in [idx_first, idx_mid, idx_last]:
        #     df2_copy = df2[(df2['df_idx'] >= idx_mid - pattern_max_width * 3) & (
        #                 df2['df_idx'] <= idx_mid + pattern_max_width * 3)].copy()
        #     df2_copy[df2_copy['df_idx'] != iii] = np.nan
        #     ap.append(mpf.make_addplot(df2_copy['typical'], type='scatter', marker='.',
        #                                markersize=200))
        #
        # data = df2[(df2['df_idx'] >= idx_mid - pattern_max_width * 3) & (
        #             df2['df_idx'] <= idx_mid + pattern_max_width * 3)]
        #
        # now = datetime.now()
        #
        #
        # mpf.plot(data, type='candle', addplot=ap,
        #          title=f"{self.ticker} {self.timeframe}",
        #          ylim=(data[['open', 'high', 'low', 'close']].min().min(),
        #                data[['open', 'high', 'low', 'close']].max().max()),
        #          savefig=f'./figures/{self.ticker}_{self.timeframe}_{now.month}_{now.day}_{now.hour}_{now.minute}.png')
        #
        # mpf.plot(data, type='candle', addplot=ap,
        #          title=f"{self.ticker} {self.timeframe}",
        #          ylim=(data[['open', 'high', 'low', 'close']].min().min(),
        #                data[['open', 'high', 'low', 'close']].max().max()))





        sys.exit()

def loop_tickers(ticker_list):
    for ticker in ticker_list:
        print(ticker)
        # for timedelta_min in range(1, 500):
        # DoubleTopBottom(
        #     ticker=ticker,
        #     timeframe='M15',
        #     local_extrema_window=3,
        #     preceding_falling_trend_pct=0.8,
        #     ATR_period=14,
        #     ATR_multiplier=1.0,
        #     pattern_max_width=50
        # ).find_double_bottom()
        # DoubleTopBottom(
        #     ticker=ticker,
        #     timeframe='M5',
        #     local_extrema_window=3,
        #     preceding_falling_trend_pct=0.8,
        #     ATR_period=14,
        #     ATR_multiplier=1.0,
        #     pattern_max_width=50
        # ).find_double_bottom()
        DoubleTopBottom(
            ticker=ticker,
            timeframe='M1',
            local_extrema_window=3,
            preceding_falling_trend_pct=0.8,
            ATR_period=14,
            MA_period=20,
            Aroon_period=20,
            ATR_multiplier=1.0,
            pattern_max_width=50
        ).find_double_bottom()

if __name__ == '__main__':

    starttime = time.time()

    # while True:

    tic = time.time()

    # with Pool(5) as p:
    #     p.map(loop_tickers, [["BTCUSD"]])
    loop_tickers(ticker_list)

    toc = time.time()
    print(f"Running at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} took {round(toc - tic,2)} seconds.")

    # time.sleep(60.0 - ((time.time() - starttime) % 60.0))

