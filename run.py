import os, math, time, json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta

# ==== CONFIG ====
USE_FIRST_N = int(os.getenv("USE_FIRST_N", "400"))   # 初期は400など軽め。慣れたら増やす
BATCH        = 150
SHORT_MA     = 25
LONG_MA      = 75
MA200        = 200
SLOPE_LB     = 5
HORIZON      = 20
TREND_LB     = 20
REQUIRE_FUNDAMENTALS = True   # 重い場合は False に

SHEET_ID     = os.environ["SHEET_ID"]  # GitHub Secrets で設定
TAB_NAME     = "watchlist"             # 公開しているタブ名

# ==== Google Sheets auth ====
import gspread
from gspread_dataframe import set_with_dataframe
sa = gspread.service_account_from_dict(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]))
sh = sa.open_by_key(SHEET_ID)

def open_or_create_tab(sheet, title):
    try:
        ws = sheet.worksheet(title)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title=title, rows=2000, cols=20)
    return ws

# ==== Load prime master (code,ticker,name) ====
pm = pd.read_csv("prime_master.csv", dtype=str).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
tickers = pm["ticker"].tolist()
if USE_FIRST_N:
    tickers = tickers[:USE_FIRST_N]
ticker_to_code = dict(zip(pm["ticker"], pm["code"]))
ticker_to_name = dict(zip(pm["ticker"], pm["name"]))

# ==== yfinance batched ====
def fetch_prices_batched(tickers, period="5y", interval="1d"):
    out = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        data = yf.download(
            tickers=batch, period=period, interval=interval,
            auto_adjust=True, group_by="ticker", threads=True, progress=False
        )
        for t in batch:
            try:
                df_t = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if df_t is not None and not df_t.empty:
                    df_t = df_t[["Open","High","Low","Close","Volume"]].dropna(how="any").copy()
                    df_t.index.name = "Date"
                    out[t] = df_t
            except Exception:
                pass
        time.sleep(0.4)
    return out

# ==== indicators ====
def add_indicators(close: pd.Series):
    s = pd.DataFrame(index=close.index)
    s["close"]   = close
    s["sma_s"]   = close.rolling(SHORT_MA).mean()
    s["sma_l"]   = close.rolling(LONG_MA).mean()
    s["sma_200"] = close.rolling(MA200).mean()
    return s

def approaching_gc(tech: pd.DataFrame):
    d = tech["sma_s"] - tech["sma_l"]
    meta = {"gap": float(d.iloc[-1]) if not d.empty else np.nan, "slope": np.nan, "proj_days": np.inf}
    if len(d.dropna()) < max(SHORT_MA, LONG_MA) + SLOPE_LB: return False, meta
    if tech["sma_s"].iloc[-1] >= tech["sma_l"].iloc[-1]:    return False, meta
    seg = d.dropna().iloc[-SLOPE_LB:]
    x = np.arange(len(seg))
    slope = np.polyfit(x, seg.values, 1)[0]
    meta["slope"] = float(slope)
    proj_days = math.inf if slope <= 0 else -d.iloc[-1] / slope
    meta["proj_days"] = float(proj_days)
    cond = (slope > 0) and (0 < proj_days <= HORIZON)
    return cond, meta

def long_term_uptrend(tech: pd.DataFrame):
    t = tech.dropna(subset=["sma_200"])
    if len(t) < TREND_LB + 1: return False
    ok_price = t["close"].iloc[-1] > t["sma_200"].iloc[-1]
    ok_slope = t["sma_200"].iloc[-1] > t["sma_200"].iloc[-1 - TREND_LB]
    return bool(ok_price and ok_slope)

# ==== fundamentals ====
def _pick_series_like(fin_df: pd.DataFrame, names):
    if fin_df is None or fin_df.empty: return None
    cols = fin_df.columns[:3] if len(fin_df.columns) >= 3 else fin_df.columns
    for name in names:
        if name in fin_df.index:
            try:
                vals = fin_df.loc[name, cols]
                vals = vals.tolist() if isinstance(vals, pd.Series) else [vals]
                vals = [float(v) for v in vals]
                return vals if len(vals) >= 3 else None
            except Exception: pass
    try:
        sub = fin_df.loc[:, cols].dropna(how="all")
        if sub.empty: return None
        s = sub.abs().mean(axis=1).sort_values(ascending=False)
        vals = fin_df.loc[s.index[0], cols]
        vals = vals.tolist() if isinstance(vals, pd.Series) else [vals]
        vals = [float(v) for v in vals]
        return vals if len(vals) >= 3 else None
    except Exception:
        return None

def three_year_growth_via_yf(ticker: str):
    if not REQUIRE_FUNDAMENTALS: return True
    try:
        tkr = yf.Ticker(ticker)
    except Exception:
        return False
    fin = None
    for attr in ("income_stmt", "financials"):
        try:
            df = getattr(tkr, attr)
            if isinstance(df, pd.DataFrame) and not df.empty:
                fin = df.copy(); break
        except Exception: continue
    if fin is None or fin.empty or len(fin.columns) < 3: return False
    revenue_candidates = ["Total Revenue","Revenue","Net Sales","NetSales","SalesRevenueNet","OperatingRevenue","Operating Revenue","NetRevenue"]
    net_income_candidates = ["Net Income","NetIncome","Net Income Common Stockholders","ProfitLoss","Profit for the year","NetIncomeToOwnersOfParent"]
    rev = _pick_series_like(fin, revenue_candidates)
    ni  = _pick_series_like(fin, net_income_candidates)
    if rev is None or ni is None: return False
    return (rev[0] > rev[1] > rev[2]) and (ni[0] > ni[1] > ni[2])

# ==== build watchlist ====
def make_today_watchlist(tickers, price_map):
    rows = []
    for t in tickers:
        df = price_map.get(t)
        if df is None or df.empty: continue
        tech = add_indicators(df["Close"])
        gc_ok, gc_meta = approaching_gc(tech)
        trend_ok = long_term_uptrend(tech)
        growth_ok = three_year_growth_via_yf(t) if REQUIRE_FUNDAMENTALS else True
        if gc_ok and trend_ok and growth_ok:
            last_dt = df.index.max()
            rows.append({
                "code": pm.loc[pm["ticker"]==t, "code"].values[0] if (pm["ticker"]==t).any() else t.replace(".T",""),
                "name": pm.loc[pm["ticker"]==t, "name"].values[0] if (pm["ticker"]==t).any() else "",
                "ticker": t,
                "last_date": last_dt.date(),
                "last_price": float(df["Close"].iloc[-1]),
                "proj_days": float(gc_meta.get("proj_days", np.inf)),
                "gc_gap": float(gc_meta.get("gap", np.nan)),
                "gc_slope": float(gc_meta.get("slope", np.nan))
            })
    if not rows:
        return pd.DataFrame(columns=["code","name","ticker","last_date","last_price","proj_days","gc_gap","gc_slope"])
    wl = pd.DataFrame(rows).sort_values(["proj_days","gc_slope"], ascending=[True, False]).reset_index(drop=True)
    return wl

# ==== run ====
price_map = fetch_prices_batched(tickers, period="5y", interval="1d")
watchlist_df = make_today_watchlist(tickers, price_map)

# ==== write to sheet ====
from gspread_dataframe import set_with_dataframe
from datetime import datetime, timezone, timedelta

# write table starting at row 2 (row 1 reserved for timestamp)
set_with_dataframe(ws, watchlist_df, row=2, col=1,
                   include_index=False, include_column_header=True, resize=True)

# timestamp in A1
JST = timezone(timedelta(hours=9))
stamp = f"Last updated (JST): {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}"
ws.update_acell('A1', stamp)   # ← NOTE: update_acell, not update("A1", "text")
print(stamp)


# stamp
JST = timezone(timedelta(hours=9))
ws.update('A1', f"Last updated (JST): {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")
print(f"Rows: {len(watchlist_df)} written to sheet '{TAB_NAME}' in {SHEET_ID}")

