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

    def __init__(self, start_dt, universe, lookbacks, z=1.0):
        super().__init__(start_dt, universe, lookbacks)
        self.history = {}
        self.ohlc = {}
        self.rsi = {}
        self.z = z

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
        return (price, average, stdev, average - self.z*stdev, average + self.z*stdev)

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

    @staticmethod
    def _adx_values(highs, lows, closes, lookback):
        """Calculate DI+, DI-, and ADX using the supplied Pine v4 formula."""
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
                high - low,
                abs(high - previous_close),
                abs(low - previous_close)
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

            smoothed_true_range = (
                smoothed_true_range
                - smoothed_true_range / lookback
                + true_range
            )
            smoothed_directional_movement_plus = (
                smoothed_directional_movement_plus
                - smoothed_directional_movement_plus / lookback
                + directional_movement_plus
            )
            smoothed_directional_movement_minus = (
                smoothed_directional_movement_minus
                - smoothed_directional_movement_minus / lookback
                + directional_movement_minus
            )

            if smoothed_true_range == 0.0:
                dx_values.append(np.nan)
                continue

            di_plus = (
                smoothed_directional_movement_plus
                / smoothed_true_range * 100.0
            )
            di_minus = (
                smoothed_directional_movement_minus
                / smoothed_true_range * 100.0
            )
            di_sum = di_plus + di_minus
            dx_values.append(
                np.nan if di_sum == 0.0
                else abs(di_plus - di_minus) / di_sum * 100.0
            )

        if len(dx_values) < lookback:
            return (np.nan, np.nan, np.nan)

        recent_dx = np.asarray(dx_values[-lookback:], dtype=float)
        if np.isnan(recent_dx).any():
            adx = np.nan
        else:
            adx = np.mean(recent_dx)

        return (di_plus, di_minus, adx)

    def _adx(self, asset, lookback):
        """Return DI+, DI-, and ADX for an asset and lookback period."""
        bars = self.ohlc.get(asset)
        if bars is None:
            return (np.nan, np.nan, np.nan)
        return self._adx_values(
            bars['high'], bars['low'], bars['close'], lookback
        )

    @staticmethod
    def _rsi_from_averages(up, down):
        """Calculate RSI from Wilder-smoothed upward and downward changes."""
        if down == 0.0:
            return 100.0
        if up == 0.0:
            return 0.0
        return 100.0 - (100.0 / (1.0 + up / down))

    def _append_rsi(self, asset, price, lookback):
        """Update Pine-style RMA gains and losses for an asset and lookback."""
        key = self._asset_lookback_key(asset, lookback)
        state = self.rsi.setdefault(key, {
            'previous_price': None,
            'gains': [],
            'losses': [],
            'raw_gains': [],
            'raw_losses': [],
            'up': None,
            'down': None,
            'raw_value': np.nan,
            'value': np.nan
        })
        previous_price = state['previous_price']
        state['previous_price'] = price
        if previous_price is None:
            return

        change = price - previous_price
        gain = max(change, 0.0)
        loss = -min(change, 0.0)
        state['raw_gains'].append(gain)
        state['raw_losses'].append(loss)
        if len(state['raw_gains']) > lookback:
            state['raw_gains'].pop(0)
            state['raw_losses'].pop(0)
        if len(state['raw_gains']) == lookback:
            state['raw_value'] = self._rsi_from_averages(
                np.mean(state['raw_gains']), np.mean(state['raw_losses'])
            )
        if state['up'] is None:
            state['gains'].append(gain)
            state['losses'].append(loss)
            if len(state['gains']) < lookback:
                return
            state['up'] = np.mean(state['gains'])
            state['down'] = np.mean(state['losses'])
            state['gains'] = []
            state['losses'] = []
        else:
            state['up'] = (state['up'] * (lookback - 1) + gain) / lookback
            state['down'] = (state['down'] * (lookback - 1) + loss) / lookback

        state['value'] = self._rsi_from_averages(
            state['up'], state['down']
        )

    def _rsi(self, asset, lookback):
        """Return the latest RSI for an asset and lookback period."""
        key = self._asset_lookback_key(asset, lookback)
        return self.rsi.get(key, {}).get('value', np.nan)

    def _raw_rsi(self, asset, lookback):
        """Return the latest unsmoothed RSI for an asset and lookback period."""
        key = self._asset_lookback_key(asset, lookback)
        return self.rsi.get(key, {}).get('raw_value', np.nan)

    def append(self, asset, price, dt=None, high=None, low=None):
        """Append a price and retain its mean-reversion values for plotting."""
        super().append(asset, price, dt)

        if (high is None) != (low is None):
            raise ValueError('Both high and low are required for ADX calculation.')
        if high is not None:
            bars = self.ohlc.setdefault(
                asset, {'high': [], 'low': [], 'close': []}
            )
            bars['high'].append(high)
            bars['low'].append(low)
            bars['close'].append(price)

        for lookback in self.lookbacks:
            key = self._asset_lookback_key(asset, lookback)
            self._append_rsi(asset, price, lookback)
            di_plus, di_minus, adx = self._adx(asset, lookback)
            self.history.setdefault(key, []).append(
                (dt,) + self._values(asset, lookback) + (
                    self._long_average(asset), di_plus, di_minus, adx,
                    self._raw_rsi(asset, lookback), self._rsi(asset, lookback)
                )
            )

    def __call__(self, asset, lookback):
        """Return price, short-window values, and the long-window average."""
        return self._values(asset, lookback) + (self._long_average(asset),)

    def get_history(self, asset, lookback):
        """Return timestamped statistics, bands, ADX, and RSI values."""
        key = self._asset_lookback_key(asset, lookback)
        return pd.DataFrame(
            self.history.get(key, []),
            columns=[
                'Date', 'Price', 'Average', 'Stdev', 'Lower Band', 'Upper Band',
                'Long Average', 'DI+', 'DI-', 'ADX', 'Raw RSI', 'RSI'
            ]
        ).set_index('Date')

    def get_adx(self, asset, lookback):
        """Return the latest DI+, DI-, and ADX values for an asset."""
        return self._adx(asset, lookback)

    def get_rsi(self, asset, lookback):
        """Return the latest RSI value for an asset and lookback period."""
        return self._rsi(asset, lookback)
