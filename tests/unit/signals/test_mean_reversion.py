from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytz

from qstrader.signals.zscore import ZScoreSignal


def test_zscore_signal_calculates_and_records_bands():
    start_dt = pd.Timestamp('2019-01-01 21:00:00', tz=pytz.utc)
    universe = Mock()
    universe.get_assets.return_value = ['EQ:SPY']
    signal = ZScoreSignal(start_dt, universe, lookback_window_size=3)

    for index, price in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        signal.append(
            'EQ:SPY', price, start_dt + pd.Timedelta(days=index)
        )

    price, average, stdev, lower_band, upper_band, zscore = signal('EQ:SPY')
    assert price == 18.0
    assert average == 16.0
    assert np.isclose(stdev, np.std([14.0, 16.0, 18.0]))
    assert np.isclose(lower_band, average - stdev)
    assert np.isclose(upper_band, average + stdev)
    assert np.isclose(zscore, (price - average) / stdev)

    history = signal.get_history('EQ:SPY')
    assert list(history.columns) == [
        'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band', 'Z-Score'
    ]
    assert history['Average'].iloc[0] != history['Average'].iloc[0]
    assert history['Price'].iloc[-1] == 18.0
    assert np.isclose(history['Upper Band'].iloc[-1], upper_band)
