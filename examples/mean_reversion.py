import sys
import pandas as pd
import pytz
from os import path

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.signals.adx import ADXSignal
from qstrader.signals.rsi import RSISignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.signals.zscore import ZScoreSignal
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

class MeanReversionAlphaModel(AlphaModel):
    def __init__(
        self, signals, short_lookback_window_size,
        long_lookback_window_size, universe
    ):
        self.signals = signals
        self.short_lookback_window_size = short_lookback_window_size
        self.long_lookback_window_size = long_lookback_window_size
        self.universe = universe
        self.target_weight = 0.0
        self.last_buy_price = 0.0
        self.was_last_stop_loss = False

    def __call__(self, dt):
        asset = self.universe.get_assets(dt)[0]

        # The signal buffers require a full lookback window before an
        # average can be calculated. Until then, remain unallocated.
        if self.signals.warmup >= max(
            self.short_lookback_window_size,
            self.long_lookback_window_size
        ):
            (
                asset_price_at_dt, avg, stdev, lower_band, upper_band,
                zscore
            ) = self.signals['zscore_short'](asset)
            long_average = self.signals['zscore_long'](asset)[1]
            raw_rsi, rsi = self.signals['rsi'](
                asset, self.short_lookback_window_size
            )
            di_plus, di_minus, adx = self.signals['adx'](
                asset, self.short_lookback_window_size
            )

            prev_weight = self.target_weight
            stop_loss_band = 2 * stdev
            stop_loss_rsi = 0   
            stop_loss_adx = 20

            loss = max(self.last_buy_price - asset_price_at_dt, 0)
            print(
                f"Price {asset_price_at_dt:.2f} z-score {zscore:.2f} "
                f"RSI {raw_rsi:.1f} <RSI> {rsi:.1f} ADX {adx:.2f} bought at "
                f"{self.last_buy_price:.2f} loss {loss:.2f}"
            )
            if (asset_price_at_dt < lower_band) and \
               (asset_price_at_dt >= long_average) and \
               (not self.was_last_stop_loss):
                self.target_weight = 1.0  # Planned buy                
            if asset_price_at_dt > upper_band:
                self.target_weight = 0.0  # Planned sell
                self.last_buy_price = 0.0

            if (asset_price_at_dt >= avg):
                self.was_last_stop_loss = False  # Price touched average : assume end of drop, buy on next low

            if (self.target_weight > prev_weight):
                self.last_buy_price = asset_price_at_dt

            if (self.target_weight > 0) and \
               ((loss > stop_loss_band) and 
                (adx > stop_loss_adx) or (rsi < stop_loss_rsi)):
                self.target_weight = 0.0    # Stop-loss sell
                self.last_buy_price = 0.0
                self.was_last_stop_loss = True

        weights = {asset: self.target_weight}
        return weights

if __name__ == "__main__":
    start_date_str = '2015-10-01'
    end_date_str = '2026-09-01'
    the_symbol = sys.argv[1]
    short_lookback_window_size = 14  # Business days
    long_lookback_window_size = 1  # Business days
    z_value = 1.0

    if (len(sys.argv) > 2):
        start_date_str = sys.argv[2]
    if (len(sys.argv) > 3):
        end_date_str = sys.argv[3]
    if (len(sys.argv) > 4):
        short_lookback_window_size = int(sys.argv[4])
    if (len(sys.argv) > 5):
        long_lookback_window_size = int(sys.argv[5])
    if (len(sys.argv) > 6):
        z_value = float(sys.argv[6])

    start_dt = pd.Timestamp(start_date_str, tz=pytz.UTC)
    end_dt = pd.Timestamp(end_date_str, tz=pytz.UTC)


    # Construct the symbols and assets necessary for the backtest
    the_symbol = sys.argv[1]
    the_eq_symbol = 'EQ:%s' % the_symbol
    strategy_symbols = [the_symbol]
    strategy_assets = [the_eq_symbol]
    strategy_universe = StaticUniverse(strategy_assets)
    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    #csv_dir = os.path.join("C:\\", "eugene", "worspace_stock", "stock", "data")
    csv_dir = "../../stock/data/";

    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols, adjust_prices=False)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    zscore_short_signal = ZScoreSignal(
        start_dt, strategy_universe, short_lookback_window_size, z=z_value
    )
    zscore_long_signal = ZScoreSignal(
        start_dt, strategy_universe, long_lookback_window_size, z=z_value
    )
    adx_signal = ADXSignal(
        start_dt, strategy_universe, lookbacks=[short_lookback_window_size]
    )
    rsi_signal = RSISignal(
        start_dt, strategy_universe, lookbacks=[short_lookback_window_size]
    )
    signals = SignalsCollection({
        'zscore_short': zscore_short_signal,
        'zscore_long': zscore_long_signal,
        'adx': adx_signal,
        'rsi': rsi_signal
    }, data_handler)

    strategy_alpha_model = MeanReversionAlphaModel(
        signals, short_lookback_window_size, long_lookback_window_size,
        strategy_universe
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
        title=(
            f'{the_symbol} mean reversion '
            f'avg={short_lookback_window_size}d/'
            f'{long_lookback_window_size}d z={z_value} stdev'
        ),
        signal_history=(
            zscore_short_signal.get_history(the_eq_symbol)
            .join(
                zscore_long_signal.get_history(the_eq_symbol)[['Average']]
                .rename(columns={'Average': 'Long Average'})
            )
            .join(
                adx_signal.get_history(
                    the_eq_symbol, short_lookback_window_size
                )
            )
            .join(
                rsi_signal.get_history(
                    the_eq_symbol, short_lookback_window_size
                )
            )
        )
    )
    tearsheet.plot_results(filename=path.join(csv_dir, f"{the_symbol}.svg"))
