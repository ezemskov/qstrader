# Mean-reversion strategy context

## Current strategy rules

The strategy in `examples/mean_reversion.py` uses a 40-business-day `MeanReversionSignal`.

- Wait until `SignalsCollection.warmup >= lookback_window_size` before calculating signals.
- `MeanReversionSignal` calculates and records the latest `price`, moving `avg`, `stdev`, `lower_band` (`avg - stdev`), and `upper_band` (`avg + stdev`).
- Buy signal: `price < avg - stdev`.
- Sell signal: `price > avg + stdev`.

## Current implementation status

`qstrader/signals/mean_reversion.py` provides `MeanReversionSignal`. It retains timestamped signal history and exposes `get_history(asset, lookback)` for plotting. `TearsheetStatistics` accepts this history through its optional `signal_history` argument and displays price, moving average, and ±1 standard-deviation bands.

`MeanReversionAlphaModel` uses this signal and retains its current target weight between buy/sell conditions.

## Why daily small orders occur

With `BacktestTradingSession(..., rebalance='daily')`, QSTrader treats a returned weight of `1.0` as a daily request to target 100% of *current total equity* in the asset.

Relevant flow:

1. `qstrader/trading/backtest.py`: at each daily market close, it calls `self.qts(dt)`.
2. `qstrader/portcon/order_sizer/dollar_weighted.py`: `DollarWeightedCashBufferedOrderSizer` converts the weight to an integer target quantity:

   ```python
   target_quantity = floor(
       total_equity * (1 - cash_buffer_percentage) / asset_price
   )
   ```

3. `qstrader/portcon/pcm.py::_generate_rebalance_orders`: compares the target quantity with current holdings and creates an order for the difference.
4. The simulated exchange considers the exact `market_close` timestamp tradable, so an order generated from a close signal executes immediately at that same close price.

Therefore, a persistent `1.0` does **not** mean “buy once and hold”. It means “rebalance daily back to 100% of equity”. Price movement changes the share target, so QSTrader submits small buy/sell adjustments.

## Same-close execution

`qstrader/exchange/simulated_exchange.py` now treats the exact 21:00 UTC daily close as a tradable instant. Therefore, the target quantity and its execution use the same close price, avoiding the prior overnight opening-price gap and associated negative-cash warnings.

This is intentionally look-ahead biased for ordinary daily OHLC data: a real strategy cannot know the final close before trading at it.

## Recommended next implementation

The desired system is stateful and should only trade on regime transitions:

| Condition | Action |
| --- | --- |
| `price < avg - stdev` while flat | Buy / allocate 100% once |
| `price > avg + stdev` while long | Sell / allocate 0% once |
| Price within the two bands | Hold current position; submit no rebalance order |

Keeping the previous target weight within the neutral band is insufficient when the session still rebalances daily: it continues to calculate a new 100%-of-equity target quantity.

A proper solution must distinguish daily signal evaluation from order generation. Add a signal-change/regime-transition mechanism and skip portfolio construction/execution when the state has not changed. This likely requires support in the portfolio-construction or rebalance layer, because `PortfolioConstructionModel` currently converts every daily weight into a target share quantity.

Conceptual alpha state:

```python
if price < lower_band and not self.in_position:
    self.in_position = True
    # Emit a buy action.
elif price > upper_band and self.in_position:
    self.in_position = False
    # Emit a sell action.
else:
    # Hold; do not submit a new target-weight rebalance.
    pass
```

A generic implementation could have the alpha model report whether its target changed, and have the trading/portfolio construction path skip order generation when it did not.
