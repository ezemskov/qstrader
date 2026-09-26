import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal


class RSISignal(Signal):
    """Raw and Wilder-smoothed relative strength index values."""

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)
        self.history = {}
        self.rsi = {}

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return '%s_%s' % (asset, lookback)

    @staticmethod
    def _rsi_from_averages(up, down):
        if down == 0.0:
            return 100.0
        if up == 0.0:
            return 0.0
        return 100.0 - (100.0 / (1.0 + up / down))

    def _values(self, asset, lookback):
        key = self._asset_lookback_key(asset, lookback)
        state = self.rsi.get(key, {})
        return (state.get('raw_value', np.nan), state.get('value', np.nan))

    def append(self, asset, price, dt=None):
        """Append a price and update raw and Wilder-smoothed RSI values."""
        super().append(asset, price, dt)
        for lookback in self.lookbacks:
            key = self._asset_lookback_key(asset, lookback)
            state = self.rsi.setdefault(key, {
                'previous_price': None, 'gains': [], 'losses': [],
                'raw_gains': [], 'raw_losses': [], 'up': None, 'down': None,
                'raw_value': np.nan, 'value': np.nan
            })
            previous_price = state['previous_price']
            state['previous_price'] = price
            if previous_price is not None:
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
                    if len(state['gains']) == lookback:
                        state['up'] = np.mean(state['gains'])
                        state['down'] = np.mean(state['losses'])
                        state['gains'] = []
                        state['losses'] = []
                else:
                    state['up'] = (state['up'] * (lookback - 1) + gain) / lookback
                    state['down'] = (state['down'] * (lookback - 1) + loss) / lookback
                if state['up'] is not None:
                    state['value'] = self._rsi_from_averages(
                        state['up'], state['down']
                    )
            self.history.setdefault(key, []).append((dt,) + self._values(asset, lookback))

    def __call__(self, asset, lookback):
        """Return raw and Wilder-smoothed RSI values."""
        return self._values(asset, lookback)

    def get_history(self, asset, lookback):
        """Return timestamped raw and Wilder-smoothed RSI values."""
        key = self._asset_lookback_key(asset, lookback)
        return pd.DataFrame(
            self.history.get(key, []), columns=['Date', 'Raw RSI', 'RSI']
        ).set_index('Date')