import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal


class ZScoreSignal(Signal):
    """Rolling price statistics, bands, and Z-scores for each asset."""

    def __init__(self, start_dt, universe, lookbacks, z=1.0):
        super().__init__(start_dt, universe, lookbacks)
        self.history = {}
        self.z = z

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return '%s_%s' % (asset, lookback)

    def _values(self, asset, lookback):
        prices = self.buffers.prices[
            self._asset_lookback_key(asset, lookback)
        ]
        if len(prices) < lookback:
            return (np.nan,) * 6

        average = np.mean(prices)
        stdev = np.std(prices)
        price = prices[-1]
        zscore = 0.0 if stdev == 0.0 else (price - average) / stdev
        return (
            price, average, stdev, average - self.z * stdev,
            average + self.z * stdev, zscore
        )

    def _long_average(self, asset):
        if len(self.lookbacks) < 2:
            return np.nan

        long_lookback = self.lookbacks[1]
        prices = self.buffers.prices[
            self._asset_lookback_key(asset, long_lookback)
        ]
        if len(prices) < long_lookback:
            return np.nan
        return np.mean(prices)

    def append(self, asset, price, dt=None):
        """Append a price and retain statistics for plotting."""
        super().append(asset, price, dt)
        for lookback in self.lookbacks:
            key = self._asset_lookback_key(asset, lookback)
            self.history.setdefault(key, []).append(
                (dt,) + self._values(asset, lookback) + (
                    self._long_average(asset),
                )
            )

    def __call__(self, asset, lookback):
        """Return price, band statistics, Z-score, and long average."""
        return self._values(asset, lookback) + (self._long_average(asset),)

    def get_history(self, asset, lookback):
        """Return timestamped price statistics and Z-score values."""
        key = self._asset_lookback_key(asset, lookback)
        return pd.DataFrame(
            self.history.get(key, []),
            columns=[
                'Date', 'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band',
                'Z-Score', 'Long Average'
            ]
        ).set_index('Date')