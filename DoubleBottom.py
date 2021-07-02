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
# ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
#                'CADCHF', 'CHFPLN', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'CHFSGD', 'AUDCAD', 'AUDJPY',
#                'AUDNZD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURNOK', 'EURNZD', 'EURSEK', 'GBPAUD', 'GBPCAD', 'GBPNZD',
#                'NZDCAD', 'NZDJPY', 'USDNOK', 'USDSEK', 'AUDDKK', 'EURHUF', 'EURMXN', 'EURPLN', 'EURTRY', 'EURZAR',
#                'GBPNOK', 'GBPPLN', 'GBPSEK', 'NOKSEK', 'PLNJPY', 'USDMXN', 'USDHUF', 'USDPLN', 'USDTRY', 'USDZAR',
#                'EURRUB', 'USDRUB', 'USDILS', 'USDCNH', 'GBPZAR', 'GOLD', 'SILVER', 'WTI', 'BRENT', 'NAT.GAS', 'GOLDoz',
#                'GOLDgr', 'GOLDEURO', 'SILVEREURO', 'AUDPLN', 'AUDSGD', 'EURSGD', 'GBPSGD', 'NZDSGD', 'USDDKK', 'SGDJPY',
#                'USDSGD', 'EURHKD', 'GBPDKK', 'USDHKD', '#AUS200', '#Swiss20', '#Spain35', '#ChinaA50', '#ChinaHShar',
#                '#Euro50', '#France120', '#France40', '#Germany30', '#Germany50', '#GerTech30', '#Holland25',
#                '#HongKong50', '#Japan225', '#UK100', '#US30', '#USNDAQ100', '#USSPX500', 'EURDKK', 'EURCZK', 'USDCZK',
#                'BTCUSD', 'ETHUSD', 'LTCUSD', 'USDTHB', 'BCHUSD', 'XRPUSD', 'PLATINUM', '#UKmid250',
#                '#UKOil_N21', '#Corn_N21', '#SBean_N21', '#Wheat_N21', '#Cotton_N21',
#                '#Sugar_N21', '#Coffee_N21', '#NGas_N21', '#USOil_N21', '#UKOil_Q21', 'DOTUSD', 'EOSUSD', 'LNKUSD',
#                'XLMUSD', 'DOGUSD', '#JP225_U21', '#DJ30_U21', '#US100_U21', '#US500_U21', '#UK100_U21', '#EUR50_U21',
#                '#GER30_U21', '#FRA40_U21', '#US$idx_U21', '#SWI20_U21', 'BATUSD', 'BTGUSD', 'DSHUSD', 'ETCUSD',
#                'IOTUSD', 'NEOUSD', 'XMRUSD', 'ZECUSD', '#Coffee_U21', '#Cocoa_U21', '#USOil_Q21', '#Corn_U21',
#                '#SBean_Q21', '#Wheat_U21', '#Cotton_V21', '#Sugar_V21', '#UKOil_U21', '#NGas_Q21', '#USOil_U21',
#                '#UKOil_V21', '#NGas_U21']

# ticker_list = ['#JP225_M21',
#                '#DJ30_M21', '#US100_M21', '#US500_M21', '#UK100_M21', '#EUR50_M21', '#GER30_M21', '#FRA40_M21',
#                '#SWI20_M21', '#NGas_M21']

ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
               'CADCHF', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'AUDCAD', 'AUDJPY', 'AUDNZD', 'CADJPY',
               'EURAUD', 'EURCAD', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDJPY', 'EURHUF', 'USDHUF',
               'EURNOK', 'USDNOK', 'EURRUB', 'USDRUB']

# ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
#                'CADCHF', 'CHFPLN', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'CHFSGD', 'AUDCAD', 'AUDJPY',
#                'AUDNZD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURNOK', 'EURNZD', 'EURSEK', 'GBPAUD', 'GBPCAD', 'GBPNZD',
#                'NZDCAD', 'NZDJPY', 'USDNOK', 'USDSEK', 'AUDDKK', 'EURHUF', 'EURMXN', 'EURPLN', 'EURTRY', 'EURZAR',
#                'GBPNOK', 'GBPPLN', 'GBPSEK', 'NOKSEK', 'PLNJPY', 'USDMXN', 'USDHUF', 'USDPLN', 'USDTRY', 'USDZAR',
#                'EURRUB', 'USDRUB', 'USDILS', 'USDCNH', 'GBPZAR', 'AUDPLN', 'AUDSGD', 'EURSGD', 'GBPSGD', 'NZDSGD',
#                'USDDKK', 'SGDJPY', 'USDSGD', 'EURHKD', 'GBPDKK', 'USDHKD', 'EURDKK', 'EURCZK', 'USDCZK',
#                'BTCUSD', 'ETHUSD', 'LTCUSD']

# ticker_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCAD', 'NZDUSD', 'AUDCHF',
#                'CADCHF', 'USDCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'NZDCHF', 'AUDCAD', 'AUDJPY', 'AUDNZD', 'CADJPY',
#                'EURAUD', 'EURCAD', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDJPY', 'EURHUF', 'USDHUF',
#                'EURNOK', 'USDNOK', 'EURRUB', 'USDRUB',
#                '#AUS200', '#Swiss20', '#Spain35', '#ChinaA50', '#ChinaHShar',
#                '#Euro50', '#France120', '#France40', '#Germany30', '#Germany50', '#GerTech30', '#Holland25',
#                '#HongKong50', '#Japan225', '#UK100', '#US30', '#USNDAQ100', '#USSPX500',
#                ]

# ticker_list = ['AUDUSD']


class DoubleTopBottom:

    def __init__(self, ticker, timeframe, local_extrema_window, preceding_falling_trend_pct, ATR_period, ATR_multiplier, pattern_max_width):
        self.ticker = ticker
        self.timeframe = timeframe
        self.local_extrema_window = local_extrema_window
        self.preceding_falling_trend_pct = preceding_falling_trend_pct
        self.ATR_period = ATR_period
        self.ATR_multiplier = ATR_multiplier
        self.pattern_max_width = pattern_max_width

        self.connect_mt5()

    def connect_mt5(self):
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

    def get_rates(self, ticker, timeframe='M1'):
        # set time zone to UTC
        # timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
        # utc_from = datetime.now()
        # print(utc_from)

        utc_from = datetime.now().astimezone(pytz.timezone("Etc/UTC"))
        # print(utc_from)

        # get 10 EURUSD M1 bars starting from 01.10.2020 in UTC time zone
        rates = mt5.copy_rates_from(ticker, eval(f"mt5.TIMEFRAME_{timeframe}"), utc_from, pattern_max_width + ATR_period)
        if rates is None:
            print(f"{ticker} doesn't exist on Mt5")
            return

        df = pd.DataFrame(columns=['<DATE>_<TIME>','open','high','low','close','tickvol','volume','spread'])
        for rate in rates:
            df = df.append(pd.Series(list(rate), index=['<DATE>_<TIME>','open','high','low','close','tickvol','volume','spread']), ignore_index=True)

        df['<DATE>_<TIME>'] = pd.to_datetime(df['<DATE>_<TIME>'],unit='s')
        df['<DATE>_<TIME>'] = df['<DATE>_<TIME>'].dt.tz_localize('Etc/UTC').dt.tz_convert('Europe/Budapest').dt.tz_localize(None)

        # df = pd.read_csv("EURUSD_M1.csv", sep="\t", parse_dates=[['<DATE>', '<TIME>']])
        # df.columns = ['<DATE>_<TIME>','open','high','low','close','tickvol','volume','spread']
        # df = df[df['<DATE>_<TIME>'] > '2021-06-07']

        df[f'ATR{ATR_period}'] = ta.ATR(df['high'], df['low'], df['close'], ATR_period)

        df['typical'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def get_minima(self):

        df = self.get_rates(self.ticker, self.timeframe)
        if df is None:
            return

        typical_prices = df['typical']

        df2 = df.set_index('<DATE>_<TIME>')[['open','high','low','close','typical',f'ATR{self.ATR_period}']]
        df2['df_idx'] = list(typical_prices.index)

        local_minima = typical_prices.iloc[argrelmin(np.array(typical_prices.values), order=local_extrema_window)].rename('Minima')
        local_maxima = typical_prices.iloc[argrelmax(np.array(typical_prices.values), order=local_extrema_window)].rename('Maxima')

        minmax_df = pd.concat([typical_prices, df[f'ATR{self.ATR_period}'],
                               pd.Series(index=typical_prices.index, name='Extremes', dtype=object)], axis=1)

        minmax_df.loc[local_minima.index, 'Extremes'] = 'L'
        minmax_df.loc[local_maxima.index, 'Extremes'] = 'H'
        minmax_df = minmax_df[~minmax_df['Extremes'].isnull()]


        # Loop through df and search for LHL or HLH combinations. Since the searched patterns are of length 3,
        # we quit the first 2 elements of the df.
        minmax_df = minmax_df.reset_index()[-4:]
        for idx in minmax_df.index[2:]:
            # Condition #0: only search for patterns if their length is max. 'pattern_max_width'
            idx_first = minmax_df.loc[idx-2,'index']
            idx_mid = minmax_df.loc[idx-1,'index']
            idx_last  = minmax_df.loc[idx,'index']
            if idx_last - idx_first > pattern_max_width:
                continue

            # Condition #1: LHL pattern
            if minmax_df.loc[idx,'Extremes'] == 'L' and minmax_df.loc[idx-1,'Extremes'] == 'H' and minmax_df.loc[idx-2,'Extremes'] == 'L':
                # Condition #2: two Ls must be approximately on same levels (within 1 ATR where ATR is calculated as the mean of
                # the ATRs of the three candles
                atr_vals_before_L = df2[(df2['df_idx'] >= max(0,idx_first - 3 * local_extrema_window)) & (df2['df_idx'] < idx_first)][f'ATR{self.ATR_period}']
                average_ATR = np.average(atr_vals_before_L)
                if average_ATR == np.nan or average_ATR is None:
                    print("\taverage_ATR is not defined for ticker: ", symbol)

                # average_ATR = np.average(minmax_df.loc[[idx-2,idx-1,idx], f'ATR{self.ATR_period}'])
                if abs(minmax_df.loc[idx,'typical']-minmax_df.loc[idx-2,'typical']) < average_ATR:
                    # Condition #3: middle peak (H) is at least N * ATR above the taller L
                    if max(minmax_df.loc[idx,'typical'],minmax_df.loc[idx-2,'typical']) + ATR_multiplier_max * average_ATR > \
                            minmax_df.loc[idx-1,'typical'] > \
                                max(minmax_df.loc[idx,'typical'],minmax_df.loc[idx-2,'typical']) + ATR_multiplier * average_ATR:
                        # Condition #4: previous N candle were mostly above the first L point
                        typical_prices_before_L = df2[(df2['df_idx'] >= idx_mid - pattern_max_width * 3) & (df2['df_idx'] < idx_first)]['typical']
                        if (typical_prices_before_L > minmax_df.loc[idx-2, 'typical']).astype(int).sum() > preceding_falling_trend_pct * len(typical_prices_before_L):
                            # Condition #5: previous 'N×local_extrema_window' candles all above the first L (this is to replicate
                            # our need to identify the first local minimum point in a skewed manner: we are much more interested
                            # to see that it is a minimum point from the left, than from the right. In other words: on the left,
                            # we want to extend the interval of investigation, we want a higher confidence that it is indeed a
                            # local minimum point, and that it is the lowest in the area.
                            typical_prices_before_L = df2[(df2['df_idx'] >= idx_first - 3*local_extrema_window) & (df2['df_idx'] < idx_first)]['typical']
                            if (typical_prices_before_L > minmax_df.loc[idx-2, 'typical']).astype(int).sum() == 3*local_extrema_window:
                                # Condition #6: last price crossed upwards through the previous highest price during the LHL formation -- buy signal
                                neckline = max(df2[(df2['df_idx'] >= idx_first) & (df2['df_idx'] <= idx_last)]['close'].max(), df2[(df2['df_idx'] >= idx_first) & (df2['df_idx'] <= idx_last)]['open'].max())
                                # print("Neckline: ", neckline)

                                # print(df2[df2['df_idx'] > idx_last]['close'] > neckline)
                                # print((df2[df2['df_idx'] > idx_last]['close'] > neckline).iloc[0])
                                # sys.exit()
                                if df2.loc[list(df2.index)[-1],'close'] > neckline > df2.loc[list(df2.index)[-2],'close']:
                                    # # Condition #7: price between last L and current candle have mostly been above the L (increasing trend)
                                    # typical_prices_after_L = df2[df2['df_idx'] > idx_last]['typical']
                                    # if (typical_prices_after_L > minmax_df.loc[idx-2, 'typical']).astype(int).sum() > .75 * len(typical_prices_after_L):

                                    # connect to the trade account without specifying a password and a server
                                    authorized = mt5.login(5386115, password="RqrOM2HP", server='FxPro-MT5')
                                    if authorized:
                                        pass
                                        # # display trading account data 'as is'
                                        # print(mt5.account_info())
                                        # # display trading account data in the form of a list
                                        # print("Show account_info()._asdict():")
                                        # account_info_dict = mt5.account_info()._asdict()
                                        # # for prop in account_info_dict:
                                        # #     print("  {}={}".format(prop, account_info_dict[prop]))
                                    else:
                                        print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))

                                    # prepare the buy request structure
                                    symbol = self.ticker
                                    symbol_info = mt5.symbol_info(symbol)
                                    if symbol_info is None:
                                        print(symbol, "not found, can not call order_check()")
                                        continue
                                        # mt5.shutdown()
                                        # quit()

                                    # if the symbol is unavailable in MarketWatch, add it
                                    if not symbol_info.visible:
                                        print(symbol, "is not visible, trying to switch on")
                                        if not mt5.symbol_select(symbol, True):
                                            print("symbol_select({}}) failed, exit", symbol)
                                            # mt5.shutdown()
                                            # quit()
                                            continue

                                    positions = mt5.positions_get(symbol=symbol)
                                    if len(positions) > 0:
                                        continue

                                    print(
                                        f"BUY OPPORTUNITY on {self.ticker} {self.timeframe} - LHL : {minmax_df.loc[idx, 'typical']:.{4}f}, {minmax_df.loc[idx - 1, 'typical']:.{4}f}, {minmax_df.loc[idx - 2, 'typical']:.{4}f}")

                                    ap = []
                                    for iii in [idx_first, idx_mid, idx_last]:
                                        df2_copy = df2[(df2['df_idx'] >= idx_mid - pattern_max_width * 3) & (
                                                    df2['df_idx'] <= idx_mid + pattern_max_width * 3)].copy()
                                        df2_copy[df2_copy['df_idx'] != iii] = np.nan
                                        ap.append(mpf.make_addplot(df2_copy['typical'], type='scatter', marker='.',
                                                                   markersize=200))

                                    data = df2[(df2['df_idx'] >= idx_mid - pattern_max_width * 3) & (
                                                df2['df_idx'] <= idx_mid + pattern_max_width * 3)]

                                    now = datetime.now()


                                    mpf.plot(data, type='candle', addplot=ap,
                                             title=f"{self.ticker} {self.timeframe}",
                                             ylim=(data[['open', 'high', 'low', 'close']].min().min(),
                                                   data[['open', 'high', 'low', 'close']].max().max()),
                                             savefig=f'./figures/{self.ticker}_{self.timeframe}_{now.month}_{now.day}_{now.hour}_{now.minute}.png')

                                    mpf.plot(data, type='candle', addplot=ap,
                                             title=f"{self.ticker} {self.timeframe}",
                                             ylim=(data[['open', 'high', 'low', 'close']].min().min(),
                                                   data[['open', 'high', 'low', 'close']].max().max()))



                                    lot = 0.1
                                    point = mt5.symbol_info(symbol).point
                                    print("Point: ", point)
                                    price = mt5.symbol_info_tick(symbol).ask
                                    spread = mt5.symbol_info(symbol).spread * point
                                    print("Spread: ", spread)
                                    stoploss = df2[(df2['df_idx'] >= idx_first) & (df2['df_idx'] <= idx_last)][
                                                   'low'].min() - spread
                                    takeprofit = price + spread * 1.5
                                    print(f"TP: {round(takeprofit, 5)}, SL: {round(stoploss, 5)}")
                                    while True:

                                        request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": symbol,
                                            "volume": lot,
                                            "type": mt5.ORDER_TYPE_BUY,
                                            "price": price,
                                            "sl": stoploss,
                                            "tp": takeprofit,
                                            # "deviation": deviation,
                                            "magic": 234000,
                                            "comment": "python script open",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC #mt5.ORDER_FILLING_RETURN,
                                        }

                                        # send a trading request
                                        result = mt5.order_send(request)

                                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                                            print(result.retcode)

                                        if result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
                                            stoploss -= 0.1 * average_ATR

                                        elif result.retcode == mt5.TRADE_RETCODE_INVALID_PRICE:
                                            takeprofit += 0.1*average_ATR

                                        elif result.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
                                            lot *= 2

                                        elif result.retcode == mt5.TRADE_RETCODE_DONE:
                                            break

                                        else:
                                            break

                                        # # check the execution result
                                        # # print("1. order_send(): by {} {} lots at {} with deviation={} points".format(
                                        # #     symbol, lot, price, deviation));
                                        # if result.retcode != mt5.TRADE_RETCODE_DONE:
                                        #     print("2. order_send failed, retcode={}".format(result.retcode))
                                        #     # request the result as a dictionary and display it element by element
                                        #     result_dict = result._asdict()
                                        #     for field in result_dict.keys():
                                        #         print("   {}={}".format(field, result_dict[field]))
                                        #         # if this is a trading request structure, display it element by element as well
                                        #         if field == "request":
                                        #             traderequest_dict = result_dict[field]._asdict()
                                        #             for tradereq_filed in traderequest_dict:
                                        #                 print("       traderequest: {}={}".format(tradereq_filed,
                                        #                                                           traderequest_dict[
                                        #                                                               tradereq_filed]))
                                        #     print("shutdown() and quit")
                                        #     continue
                                        #     # mt5.shutdown()
                                        #     # quit()

                                    # print("2. order_send done, ", result)
                                    # print("   opened position with POSITION_TICKET={}".format(result.order))
                                    # # print("   sleep 2 seconds before closing position #{}".format(result.order))
                                    # time.sleep(2)
                                    # # create a close request
                                    # position_id = result.order
                                    # price = mt5.symbol_info_tick(symbol).bid
                                    # # deviation = 20
                                    # request = {
                                    #     "action": mt5.TRADE_ACTION_DEAL,
                                    #     "symbol": symbol,
                                    #     "volume": lot,
                                    #     "type": mt5.ORDER_TYPE_SELL,
                                    #     "position": position_id,
                                    #     "price": price,
                                    #     #"deviation": deviation,
                                    #     "magic": 234000,
                                    #     "comment": "python script close",
                                    #     "type_time": mt5.ORDER_TIME_GTC,
                                    #     "type_filling": mt5.ORDER_FILLING_RETURN,
                                    # }
                                    # # send a trading request
                                    # result = mt5.order_send(request)
                                    # # check the execution result
                                    # # print(
                                    # #     "3. close position #{}: sell {} {} lots at {} with deviation={} points".format(
                                    # #         position_id, symbol, lot, price, deviation));
                                    # if result.retcode != mt5.TRADE_RETCODE_DONE:
                                    #     print("4. order_send failed, retcode={}".format(result.retcode))
                                    #     print("   result", result)
                                    # else:
                                    #     print("4. position #{} closed, {}".format(position_id, result))
                                    #     # request the result as a dictionary and display it element by element
                                    #     result_dict = result._asdict()
                                    #     for field in result_dict.keys():
                                    #         print("   {}={}".format(field, result_dict[field]))
                                    #         # if this is a trading request structure, display it element by element as well
                                    #         if field == "request":
                                    #             traderequest_dict = result_dict[field]._asdict()
                                    #             for tradereq_filed in traderequest_dict:
                                    #                 print("       traderequest: {}={}".format(tradereq_filed,
                                    #                                                           traderequest_dict[
                                    #                                                               tradereq_filed]))


def loop_tickers(ticker_list):
    for ticker in ticker_list:
        print(ticker)
        # DoubleTopBottom(
        #     ticker=ticker,
        #     timeframe='M15',
        #     local_extrema_window=3,
        #     preceding_falling_trend_pct=0.8,
        #     ATR_period=14,
        #     ATR_multiplier=1.0,
        #     pattern_max_width=50
        # ).get_minima()
        # DoubleTopBottom(
        #     ticker=ticker,
        #     timeframe='M5',
        #     local_extrema_window=3,
        #     preceding_falling_trend_pct=0.8,
        #     ATR_period=14,
        #     ATR_multiplier=1.0,
        #     pattern_max_width=50
        # ).get_minima()
        DoubleTopBottom(
            ticker=ticker,
            timeframe='M1',
            local_extrema_window=3,
            preceding_falling_trend_pct=0.8,
            ATR_period=14,
            ATR_multiplier=1.0,
            pattern_max_width=50
        ).get_minima()

if __name__ == '__main__':

    starttime = time.time()

    while True:

        tic = time.time()

        with Pool(5) as p:
            p.map(loop_tickers, [ticker_list])

        toc = time.time()
        print(f"Running at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} took {round(toc - tic,2)} seconds.")

        time.sleep(60.0 - ((time.time() - starttime) % 60.0))

