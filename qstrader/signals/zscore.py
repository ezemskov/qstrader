import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal


class ZScoreSignal(Signal):
    """Rolling price statistics, bands, and Z-scores for each asset."""

    def __init__(self, start_dt, universe, lookback_window_size, z=1.0):
        super().__init__(start_dt, universe, [lookback_window_size])
        self.history = {}
        self.lookback_window_size = lookback_window_size
        self.z = z

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return '%s_%s' % (asset, lookback)

    def _values(self, asset):
        lookback = self.lookback_window_size
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

    def append(self, asset, price, dt=None):
        """Append a price and retain statistics for plotting."""
        super().append(asset, price, dt)
        key = self._asset_lookback_key(asset, self.lookback_window_size)
        self.history.setdefault(key, []).append((dt,) + self._values(asset))

    def __call__(self, asset):
        """Return price, band statistics, and Z-score."""
        return self._values(asset)

    def get_history(self, asset):
        """Return timestamped price statistics and Z-score values."""
        key = self._asset_lookback_key(asset, self.lookback_window_size)
        return pd.DataFrame(
            self.history.get(key, []),
            columns=[
                'Date', 'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band',
                'Z-Score'
            ]
        ).set_index('Date')