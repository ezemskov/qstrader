import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal


class MeanReversionSignal(Signal):
    """Rolling price statistics and bands for a mean-reversion strategy.

    For each asset and lookback period, this signal calculates the current
    price, simple moving average, population standard deviation, and one
    standard-deviation upper and lower bands. It also retains the calculations
    with their timestamps for plotting after a backtest.
    """

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)
        self.history = {}

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return '%s_%s' % (asset, lookback)

    def _values(self, asset, lookback):
        prices = self.buffers.prices[
            self._asset_lookback_key(asset, lookback)
        ]
        if len(prices) < lookback:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        average = np.mean(prices)
        stdev = np.std(prices)
        price = prices[-1]
        return (price, average, stdev, average - stdev, average + stdev)

    def append(self, asset, price, dt=None):
        """Append a price and retain its mean-reversion values for plotting."""
        super().append(asset, price, dt)

        for lookback in self.lookbacks:
            key = self._asset_lookback_key(asset, lookback)
            self.history.setdefault(key, []).append(
                (dt,) + self._values(asset, lookback)
            )

    def __call__(self, asset, lookback):
        """Return ``(price, average, stdev, lower_band, upper_band)``."""
        return self._values(asset, lookback)

    def get_history(self, asset, lookback):
        """Return timestamped price, average, deviation, and band values."""
        key = self._asset_lookback_key(asset, lookback)
        return pd.DataFrame(
            self.history.get(key, []),
            columns=['Date', 'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band']
        ).set_index('Date')
