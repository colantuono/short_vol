import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd

## https://www.mql5.com/en/docs/python_metatrader5

## read credentials
login, passw = open('credentials').read().split()

## open session with metatrade5
mt5.initialize(login=int(login),
               password=passw,
               server='ClearInvestimentos-CLEAR')

## get tick data
def tick_data(ticker: str, from_date: datetime, to_date:datetime, info = mt5.COPY_TICKS_ALL) -> pd.DataFrame:
    ticks = mt5.copy_ticks_range("BOVA11",
                                from_date,
                                to_date,
                                info)

    df_ticks = pd.DataFrame(ticks)

    df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
    df_ticks['time_msc'] = pd.to_datetime(df_ticks['time_msc'], unit='ms')

    return df_ticks

## get candle date
def candle_data(ticker: str, from_date: datetime, to_date: datetime, timeframe = mt5.TIMEFRAME_D1) -> pd.DataFrame:
    candles = mt5.copy_rates_range(ticker,
                        timeframe,
                        from_date,
                        to_date)

    df_candles = pd.DataFrame(candles)

    df_candles['time'] = pd.to_datetime(df_candles['time'], unit='s')
    
    return df_candles
