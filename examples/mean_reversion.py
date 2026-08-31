import os

import pandas as pd
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.signals.sma import SMASignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

class MeanReversionAlphaModel(AlphaModel):
    def __init__(self, signals, lookback_window_size, universe):
        self.signals = signals
        self.lookback_window_size = lookback_window_size
        self.universe = universe

    def __call__(self, dt):
        asset = self.universe.get_assets(dt)[0]
        weights = {asset: 0.0}

        # The signal buffers require a full lookback window before an
        # average can be calculated. Until then, remain unallocated.
        if self.signals.warmup < self.lookback_window_size:
            return weights

        avg = self.signals['sma'](asset, self.lookback_window_size)
        if avg < 5:
            weights[asset] = 1.0

        return weights

if __name__ == "__main__":
    start_dt = pd.Timestamp('2025-01-31 10:00:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2026-08-01 10:00:00', tz=pytz.UTC)

    lookback_window_size = 40  # Business days

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = ['MEZ']
    strategy_assets = ['EQ:%s' % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    signal = SMASignal(start_dt, strategy_universe, lookbacks=[lookback_window_size])
    signals = SignalsCollection({'sma': signal}, data_handler)

    strategy_alpha_model = MeanReversionAlphaModel(
        signals, lookback_window_size, strategy_universe
    )
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        rebalance='daily',
        long_only=True,
        cash_buffer_percentage=0.1,
        initial_cash=20000,
        data_handler=data_handler
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_assets = ['EQ:MEZ']
    benchmark_universe = StaticUniverse(benchmark_assets)

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:MEZ': 1.0})
    benchmark_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=data_handler
    )
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='MEZ mean reversion'
    )
    tearsheet.plot_results()
