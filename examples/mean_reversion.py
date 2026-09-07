import os

import pandas as pd
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.signals.mean_reversion import MeanReversionSignal
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
        self.target_weight = 0.0
        self.last_buy_price = 0.0

    def __call__(self, dt):
        asset = self.universe.get_assets(dt)[0]

        # The signal buffers require a full lookback window before an
        # average can be calculated. Until then, remain unallocated.
        if self.signals.warmup >= self.lookback_window_size:
            asset_price_at_dt, avg, stdev, lower_band, upper_band = \
                self.signals['mean_reversion'](
                    asset, self.lookback_window_size
                )

            prev_weight = self.target_weight
            stop_loss_band = 2 * stdev

            loss = max(self.last_buy_price - asset_price_at_dt, 0)
            print(f"Price {asset_price_at_dt:.2f} avg {avg:.2f} stdev {stdev:.2f} bought at {self.last_buy_price:.2f} loss {loss:.2f}")
            if asset_price_at_dt < lower_band:
                self.target_weight = 1.0  # Planned buy                
            if asset_price_at_dt > upper_band:
                self.target_weight = 0.0  # Planned sell

            if (self.target_weight > prev_weight):
                self.last_buy_price = asset_price_at_dt

            if (self.target_weight > 0) and (loss > stop_loss_band):
                self.target_weight = 0.0    # Stop-loss sell

        weights = {asset: self.target_weight}
        return weights

if __name__ == "__main__":
    start_dt = pd.Timestamp('2025-01-01 10:00:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2026-08-01 10:00:00', tz=pytz.UTC)

    lookback_window_size = 20  # Business days

    # Construct the symbols and assets necessary for the backtest
    the_symbol = 'GNE'
    the_eq_symbol = 'EQ:%s' % the_symbol
    strategy_symbols = [the_symbol]
    strategy_assets = [the_eq_symbol]
    strategy_universe = StaticUniverse(strategy_assets)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    #csv_dir = os.path.join("C:\\", "eugene", "worspace_stock", "stock", "data")
    csv_dir = "c:/eugene/worspace_stock/stock/data/";
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    signal = MeanReversionSignal(
        start_dt, strategy_universe, lookbacks=[lookback_window_size], z=1.0
    )
    signals = SignalsCollection({'mean_reversion': signal}, data_handler)

    strategy_alpha_model = MeanReversionAlphaModel(
        signals, lookback_window_size, strategy_universe
    )
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance='daily',
        long_only=True,
        cash_buffer_percentage=0.0,
        initial_cash=10000,
        data_handler=data_handler
    )
    strategy_backtest.run()

    # Construct benchmark assets
    benchmark_universe = StaticUniverse(strategy_assets)

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({the_eq_symbol: 1.0})
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
        title=f'{the_symbol} mean reversion',
        signal_history=signal.get_history(the_eq_symbol, lookback_window_size)
    )
    tearsheet.plot_results()
