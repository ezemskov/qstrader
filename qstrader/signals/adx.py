import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal


class ADXSignal(Signal):
    """Directional indicators and ADX calculated from optional OHLC bars."""

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)
        self.history = {}
        self.ohlc = {}

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return '%s_%s' % (asset, lookback)

    @staticmethod
    def _adx_values(highs, lows, closes, lookback):
        if len(highs) < lookback or len(highs) != len(lows):
            return (np.nan, np.nan, np.nan)

        smoothed_true_range = 0.0
        smoothed_directional_movement_plus = 0.0
        smoothed_directional_movement_minus = 0.0
        dx_values = []

        for index, (high, low) in enumerate(zip(highs, lows)):
            previous_high = highs[index - 1] if index else 0.0
            previous_low = lows[index - 1] if index else 0.0
            previous_close = closes[index - 1] if index else 0.0
            true_range = max(
                high - low, abs(high - previous_close), abs(low - previous_close)
            )
            upward_movement = high - previous_high
            downward_movement = previous_low - low
            directional_movement_plus = (
                max(upward_movement, 0.0)
                if upward_movement > downward_movement else 0.0
            )
            directional_movement_minus = (
                max(downward_movement, 0.0)
                if downward_movement > upward_movement else 0.0
            )
            smoothed_true_range += true_range - smoothed_true_range / lookback
            smoothed_directional_movement_plus += (
                directional_movement_plus
                - smoothed_directional_movement_plus / lookback
            )
            smoothed_directional_movement_minus += (
                directional_movement_minus
                - smoothed_directional_movement_minus / lookback
            )
            if smoothed_true_range == 0.0:
                dx_values.append(np.nan)
                continue
            di_plus = smoothed_directional_movement_plus / smoothed_true_range * 100.0
            di_minus = smoothed_directional_movement_minus / smoothed_true_range * 100.0
            di_sum = di_plus + di_minus
            dx_values.append(
                np.nan if di_sum == 0.0
                else abs(di_plus - di_minus) / di_sum * 100.0
            )

        recent_dx = np.asarray(dx_values[-lookback:], dtype=float)
        adx = np.nan if np.isnan(recent_dx).any() else np.mean(recent_dx)
        return (di_plus, di_minus, adx)

    def _values(self, asset, lookback):
        bars = self.ohlc.get(asset)
        if bars is None:
            return (np.nan, np.nan, np.nan)
        return self._adx_values(bars['high'], bars['low'], bars['close'], lookback)

    def append(self, asset, price, dt=None, high=None, low=None):
        """Append a close and optional high/low values for ADX calculation."""
        super().append(asset, price, dt)
        if (high is None) != (low is None):
            raise ValueError('Both high and low are required for ADX calculation.')
        if high is None:
            high = price
            low = price
        bars = self.ohlc.setdefault(
            asset, {'high': [], 'low': [], 'close': []}
        )
        bars['high'].append(high)
        bars['low'].append(low)
        bars['close'].append(price)

        for lookback in self.lookbacks:
            key = self._asset_lookback_key(asset, lookback)
            self.history.setdefault(key, []).append((dt,) + self._values(asset, lookback))

    def __call__(self, asset, lookback):
        """Return DI+, DI-, and ADX for the asset and lookback."""
        return self._values(asset, lookback)

    def get_history(self, asset, lookback):
        """Return timestamped DI+, DI-, and ADX values."""
        key = self._asset_lookback_key(asset, lookback)
        return pd.DataFrame(
            self.history.get(key, []), columns=['Date', 'DI+', 'DI-', 'ADX']
        ).set_index('Date')