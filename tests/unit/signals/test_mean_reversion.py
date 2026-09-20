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
        'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band', 'Long Average'
    ]
    assert history['Average'].iloc[0] != history['Average'].iloc[0]
    assert history['Price'].iloc[-1] == 18.0
    assert np.isclose(history['Upper Band'].iloc[-1], upper_band)
    assert np.isclose(history['Long Average'].iloc[-1], 14.0)
