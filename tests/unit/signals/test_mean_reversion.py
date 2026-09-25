from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytz

from qstrader.signals.mean_reversion import MeanReversionSignal


def test_mean_reversion_signal_calculates_and_records_bands():
    start_dt = pd.Timestamp('2019-01-01 21:00:00', tz=pytz.utc)
    universe = Mock()
    universe.get_assets.return_value = ['EQ:SPY']
    signal = MeanReversionSignal(start_dt, universe, lookbacks=[3, 5])

    for index, price in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        signal.append(
            'EQ:SPY', price, start_dt + pd.Timedelta(days=index)
        )

    price, average, stdev, lower_band, upper_band, long_average = signal('EQ:SPY', 3)
    assert price == 18.0
    assert average == 16.0
    assert np.isclose(stdev, np.std([14.0, 16.0, 18.0]))
    assert np.isclose(lower_band, average - stdev)
    assert np.isclose(upper_band, average + stdev)
    assert np.isclose(long_average, 14.0)

    history = signal.get_history('EQ:SPY', 3)
    assert list(history.columns) == [
        'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band', 'Long Average',
        'DI+', 'DI-', 'ADX', 'Raw RSI', 'RSI'
    ]
    assert history['Average'].iloc[0] != history['Average'].iloc[0]
    assert history['Price'].iloc[-1] == 18.0
    assert np.isclose(history['Upper Band'].iloc[-1], upper_band)
    assert np.isclose(history['Long Average'].iloc[-1], 14.0)


def test_mean_reversion_signal_calculates_pine_rsi():
    start_dt = pd.Timestamp('2019-01-01 21:00:00', tz=pytz.utc)
    universe = Mock()
    universe.get_assets.return_value = ['EQ:SPY']
    signal = MeanReversionSignal(start_dt, universe, lookbacks=[3])

    for index, price in enumerate([10.0, 12.0, 11.0, 14.0, 13.0]):
        signal.append(
            'EQ:SPY', price, start_dt + pd.Timedelta(days=index)
        )

    # Initial RMA values are 5 / 3 and 1 / 3. The final changes update them
    # to 10 / 9 and 5 / 9, yielding Pine's RSI formula result of 200 / 3.
    assert np.isclose(signal.get_rsi('EQ:SPY', 3), 200.0 / 3.0)

    history = signal.get_history('EQ:SPY', 3)
    assert np.isnan(history['Raw RSI'].iloc[2])
    assert np.isclose(history['Raw RSI'].iloc[-1], 60.0)
    assert np.isnan(history['RSI'].iloc[2])
    assert np.isclose(history['RSI'].iloc[-1], 200.0 / 3.0)


def test_mean_reversion_signal_calculates_pine_adx_from_ohlc():
    start_dt = pd.Timestamp('2019-01-01 21:00:00', tz=pytz.utc)
    universe = Mock()
    universe.get_assets.return_value = ['EQ:SPY']
    signal = MeanReversionSignal(start_dt, universe, lookbacks=[3])

    bars = [(10.0, 9.0, 9.5), (11.0, 10.0, 10.5), (12.0, 11.0, 11.5)]
    for index, (high, low, close) in enumerate(bars):
        signal.append(
            'EQ:SPY', close, start_dt + pd.Timedelta(days=index),
            high=high, low=low
        )

    di_plus, di_minus, adx = signal.get_adx('EQ:SPY', 3)
    assert di_plus > 0.0
    assert di_minus == 0.0
    assert adx == 100.0

    history = signal.get_history('EQ:SPY', 3)
    assert {'DI+', 'DI-', 'ADX'}.issubset(history.columns)
    assert history['ADX'].iloc[-1] == adx
