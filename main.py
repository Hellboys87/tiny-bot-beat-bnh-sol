"""Dip-buying mean-reversion bot for the make.fun bars-replay/v1 harness.

Strategy (paper trading, hourly bars, 7 Solana symbols, 10,000 USDC):
  * Mean reversion: buy a symbol when it is oversold -- its close is DIP below
    its 12h EMA and its 6h return is below R6 (a sharp short dip inside a
    choppy market). A second trigger catches slow drifts the sharp-dip
    trigger misses: close DRIFT below the EMA with a flat 6h return (price
    grinding below its mean in a flat market rather than dropping sharply)
    -- these add trades in ranging conditions where the sharp dip rarely
    fires, which matters on the short 72-bar final window. The same exits
    (stop, trailing stop, time exit) bound the downside of these entries.
  * Exits: take profit at +BOUNCE from entry, hard stop at -STOP, a trailing
    stop TRAIL below the peak since entry, or a time exit after MAX_HOLD bars
    (stale positions are sold to free capital for fresh dips). No shorts,
    no leverage.
  * Sizing: up to SIZE of equity per position (capped at the 25% max weight),
    at most MAX_POS open positions, buys in one reply capped at 95% of cash
    (a sell's proceeds never fund a buy in the same reply).
  * At most one order per symbol per bar; orders decided on bar k fill at
    bar k+1's open, so every signal uses only closed bars (no look-ahead).
  * The first MIN_HIST bars only build history (the 12h EMA needs time to
    settle; trading on a noisy early EMA buys falling knives).
  * A momentum-breakout trigger (MOM) is built in but disabled by default:
    buying 12h breakouts lost money in backtests on this market, which
    mean-reverts on hourly scales.

The bot is fully deterministic: it is a pure function of the hello/batch
messages, uses no clock, randomness, environment or network, and writes
nothing to disk. Diagnostics go to stderr only; stdout carries replies.

Parameters (defaults are the tuned variant; see README "Run it"):
  DIP 0.015, R6 -0.01, DRIFT 0.01, MOM 9.9 (breakout trigger off), BOUNCE 0.08,
  STOP 0.05, TRAIL 0.08, MAX_HOLD 30, SIZE 0.25, MAX_POS 5, EMA_SLOW 12,
  MIN_HIST 26.
"""

import json
import sys

# --- strategy parameters -------------------------------------------------
DIP = 0.015        # entry: close this far below the 12h EMA (fraction)
R6 = -0.01         # entry: 6h return must be below this
DRIFT = 0.01       # slow-drift entry: close this far below the 12h EMA
                   # with a flat 6h return (see decide)
MOM_N = 12         # momentum lookback (bars) for the breakout trigger
MOM = 9.9          # entry: 12h return above this + close above EMA12 (9.9 = momentum trigger off)
BOUNCE = 0.08      # exit: take profit at this return from entry
STOP = 0.05        # exit: hard stop at this loss from entry
TRAIL = 0.08       # exit: trailing stop this far below the peak since entry
MAX_HOLD = 30      # exit: sell a position held this many bars without a signal
SIZE = 0.25        # target weight per position (fraction of equity)
MAX_POS = 5        # max open positions held
EMA_SLOW = 12      # slow EMA span (bars)
MIN_HIST = 26      # bars of history needed before trading
MAX_SLIPPAGE_BPS = 1000  # 10%: avoid rejections on hourly gaps + impact
MAX_ORDERS = 16    # harness cap per bar (checked in hello, used as a guard)


def ema_update(prev, price, span):
    """One EMA step on raw price units (ratios are scale-free)."""
    k = 2.0 / (span + 1)
    return price * k + prev * (1.0 - k) if prev is not None else float(price)


class Bot:
    def __init__(self, hello):
        self.symbols = [s["symbol"] for s in hello["symbols"]]
        self.scales = {
            s["symbol"]: (s["price_scale"], s["qty_scale"]) for s in hello["symbols"]
        }
        self.quote_scale = hello["capital"]["quote_scale"]
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        self.max_weight_bps = hello["caps"]["max_weight_bps"]
        self.max_pos = min(MAX_POS, hello["caps"]["max_open_positions"])
        self.max_orders = hello["caps"]["max_orders_per_bar"]
        self.warmup_bars = hello.get("warmup_bars", 0)
        self.closes = {s: [] for s in self.symbols}   # raw close ints
        self.ema12 = {s: None for s in self.symbols}  # slow EMA of closes
        self.entry = {}   # symbol -> entry price (raw units, from fills)
        self.peak = {}    # symbol -> highest close since entry
        self.since = {}   # symbol -> batch seq of the buy fill
        self.seq = -1
        self.next_id = 1

    def client_id(self):
        cid = "o%d" % self.next_id
        self.next_id += 1
        return cid

    def position_value(self, symbol, qty, price):
        """Value of qty at price, in quote units (ints)."""
        ps, qs = self.scales[symbol]
        return qty * price * (10 ** self.quote_scale) // (10 ** (qs + ps))

    def on_batch(self, batch):
        self.seq = batch["seq"]
        for bar in batch["bars"]:
            sym = bar["symbol"]
            if sym not in self.closes:
                continue
            c = int(bar["c"])
            self.closes[sym].append(c)
            self.ema12[sym] = ema_update(self.ema12[sym], c, EMA_SLOW)
        # Fold fills into entry/peak bookkeeping (account is authoritative
        # for what is actually held; fills tell us at what price we got in).
        for fill in batch.get("fills", []):
            if fill.get("status") != "filled":
                continue
            sym = fill["symbol"]
            price = int(fill["price"])
            if fill["side"] == "buy":
                qty = int(fill["qty"])
                if sym in self.entry:
                    old_qty = self._held_qty(sym)
                    if old_qty > 0:
                        self.entry[sym] = (
                            self.entry[sym] * old_qty + price * qty
                        ) // (old_qty + qty)
                    else:
                        self.entry[sym] = price
                else:
                    self.entry[sym] = price
                self.peak[sym] = max(self.peak.get(sym, price), price)
                self.since[sym] = self.seq
            else:
                # A sell removes the position (we always sell in full).
                self.entry.pop(sym, None)
                self.peak.pop(sym, None)
                self.since.pop(sym, None)
        account = batch["account"]
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        held = {}
        for p in account["positions"]:
            held[p["symbol"]] = int(p["qty"])
        self._held = held
        # Fallback entry price if a position exists with no fill record
        # (e.g. resumed state): use cost / qty.
        for p in account["positions"]:
            sym = p["symbol"]
            if sym not in self.entry and int(p["qty"]) > 0:
                self.entry[sym] = int(p["cost_quote_units"]) * 10 ** (
                    self.scales[sym][1] + self.scales[sym][0]
                ) // (int(p["qty"]) * 10 ** self.quote_scale) or int(
                    self.closes[sym][-1]
                )
                self.peak[sym] = self.entry[sym]
                self.since.setdefault(sym, self.seq)
        if self.seq < self.warmup_bars:
            return []
        return self.decide(held)

    def _held_qty(self, symbol):
        return getattr(self, "_held", {}).get(symbol, 0)

    def decide(self, held):
        orders = []
        cash_left = int(self.cash * 0.95)
        # --- exits first -------------------------------------------------
        sold = set()
        for sym, qty in held.items():
            if qty <= 0 or sym not in self.closes or not self.closes[sym]:
                continue
            c = self.closes[sym][-1]
            entry = self.entry.get(sym)
            if entry is None or entry <= 0:
                continue
            self.peak[sym] = max(self.peak.get(sym, entry), c)
            ret = c / entry - 1.0
            stale = self.seq - self.since.get(sym, self.seq) >= MAX_HOLD
            if (
                ret <= -STOP
                or ret >= BOUNCE
                or c < self.peak[sym] * (1.0 - TRAIL)
                or stale
            ):
                orders.append(
                    {
                        "side": "sell",
                        "client_id": self.client_id(),
                        "symbol": sym,
                        "qty": str(qty),
                        "max_slippage_bps": MAX_SLIPPAGE_BPS,
                    }
                )
                sold.add(sym)
        # --- entries ------------------------------------------------------
        slots = self.max_pos - len(held) + len(sold)
        if slots > 0 and len(self.closes[self.symbols[0]]) >= MIN_HIST:
            cands = []
            for sym in self.symbols:
                if sym in held:
                    continue  # held, or just sold this bar: never re-buy same bar
                hist = self.closes[sym]
                if len(hist) < MIN_HIST or self.ema12[sym] is None:
                    continue
                c = hist[-1]
                dip = (self.ema12[sym] - c) / self.ema12[sym]
                ret6 = c / hist[-7] - 1.0 if len(hist) >= 7 else 0.0
                mom = c / hist[-MOM_N - 1] - 1.0 if len(hist) > MOM_N else 0.0
                score = None
                if dip > DIP and ret6 < R6:
                    score = dip
                elif dip > DRIFT and R6 <= ret6 < 0.01:
                    # Slow drift below the EMA with a flat 6h return:
                    # ranging-market mean reversion the sharp-dip
                    # trigger misses. No uptick filter: in a flat
                    # market the drift itself is the signal, and the
                    # stop / trailing stop / time exit bound the risk.
                    score = dip
                elif mom > MOM and c > self.ema12[sym]:
                    score = mom
                if score is not None:
                    cands.append((score, sym))
            cands.sort(reverse=True)
            for _score, sym in cands[:slots]:
                if len(orders) >= self.max_orders:
                    break
                room = self.max_weight_bps * self.equity // 10000
                notional = min(int(self.equity * SIZE), room, cash_left)
                if notional < 1_000_000:  # below the 1 USDC min notional
                    continue
                orders.append(
                    {
                        "side": "buy",
                        "client_id": self.client_id(),
                        "symbol": sym,
                        "notional_quote_units": str(notional),
                        "max_slippage_bps": MAX_SLIPPAGE_BPS,
                    }
                )
                cash_left -= notional
        return orders[: self.max_orders]


def main():
    bot = None
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            bot = Bot(msg)
        elif kind == "batch":
            orders = bot.on_batch(msg)
            out.write(
                json.dumps(
                    {"type": "reply", "seq": msg["seq"], "orders": orders},
                    separators=(",", ":"),
                )
                + "\n"
            )
            out.flush()
        elif kind == "done":
            break


if __name__ == "__main__":
    main()
