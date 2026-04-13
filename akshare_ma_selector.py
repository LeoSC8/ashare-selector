"""基于 AKShare 的 A 股开盘均线三档信号筛选器。

信号定义：
1) Cross Up
2) Near Cross
3) Hold After Cross

使用说明（示例）：
    # 1) 默认读取 selector_config.json
    python akshare_ma_selector.py

    # 2) 命令行参数优先于配置文件
    python akshare_ma_selector.py --symbols 000001,600519,300750 --x 0.0025
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "未安装 akshare，请先执行: pip install akshare pandas"
    ) from exc

try:
    import requests
    from curl_cffi import requests as curl_requests
    import time as _time

    def _curl_get(url, **kwargs):
        kwargs.pop("timeout", None)
        for _attempt in range(3):
            try:
                _time.sleep(0.3)  # 请求间短暂延迟，避免被限流
                return curl_requests.get(url, impersonate="chrome", timeout=30, **kwargs)
            except Exception:
                if _attempt == 2:
                    raise
                _time.sleep(2)  # 失败后等待更久再重试

    requests.get = _curl_get
except ImportError:
    pass


@dataclass
class Thresholds:
    """阈值配置（全部使用小数表示百分比）。"""

    x: float = 0.003  # 0.3%
    y: float = 0.0008  # 0.08%
    r: float = 0.9  # 90%
    open_limit: float = 0.01  # 1%


@dataclass
class SignalResult:
    symbol: str
    tier: int
    tier_name: str
    open_t: float
    close_t_1: float
    ma5_prev: float
    ma10_prev: float
    ma5_now: float
    ma10_now: float
    a5: float
    a10: float
    gap_prev_pct: float
    gap_now_pct: float
    open_dev: float
    score: float
    risk_flags: str


def _parse_trade_date(value: Optional[str]) -> date:
    if not value:
        return datetime.utcnow().date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def _build_spot_snapshot() -> pd.DataFrame:
    """拉取 A 股实时快照，提取代码、今开、昨收。"""
    spot = ak.stock_zh_a_spot_em()
    required = {"代码", "今开", "昨收"}
    missing = required.difference(spot.columns)
    if missing:
        raise RuntimeError(f"实时行情缺少字段: {missing}")
    out = spot[["代码", "今开", "昨收"]].copy()
    out["代码"] = out["代码"].astype(str).str.zfill(6)
    out["今开"] = pd.to_numeric(out["今开"], errors="coerce")
    out["昨收"] = pd.to_numeric(out["昨收"], errors="coerce")
    return out


def _get_open_and_prev_close(symbol: str, asof: date, spot_df: Optional[pd.DataFrame]) -> Optional[tuple[float, float]]:
    """获取指定交易日的 open_t 与 close_t_1。

    - 当 asof >= 今天：优先使用实时快照中的 今开/昨收。
    - 当 asof < 今天：使用历史日线，取 asof 当天开盘 和 前一交易日收盘。
    """
    today = datetime.utcnow().date()
    if asof >= today:
        if spot_df is None or symbol not in spot_df.index:
            return None
        open_t = spot_df.at[symbol, "今开"]
        close_t_1 = spot_df.at[symbol, "昨收"]
        if pd.isna(open_t) or pd.isna(close_t_1):
            return None
        return float(open_t), float(close_t_1)

    start = (asof - timedelta(days=30)).strftime("%Y%m%d")
    end = asof.strftime("%Y%m%d")
    hist = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if hist is None or hist.empty:
        return None
    if "日期" not in hist.columns or "开盘" not in hist.columns or "收盘" not in hist.columns:
        return None

    h = hist[["日期", "开盘", "收盘"]].copy()
    h["日期"] = pd.to_datetime(h["日期"]).dt.date
    h["开盘"] = pd.to_numeric(h["开盘"], errors="coerce")
    h["收盘"] = pd.to_numeric(h["收盘"], errors="coerce")
    h = h.dropna(subset=["开盘", "收盘"]).sort_values("日期")
    h = h[h["日期"] <= asof]
    if len(h) < 2:
        return None

    # asof 当天必须有日线记录，上一行作为 close_t_1
    today_rows = h[h["日期"] == asof]
    if today_rows.empty:
        return None
    idx = today_rows.index[-1]
    pos = h.index.get_indexer([idx])[0]
    if pos <= 0:
        return None
    open_t = float(h.iloc[pos]["开盘"])
    close_t_1 = float(h.iloc[pos - 1]["收盘"])
    return open_t, close_t_1


def _get_recent_closes(symbol: str, asof: date, bars: int = 10) -> Optional[List[float]]:
    """获取指定日期前最近 bars 个日线收盘价（不含当日）。"""
    start = (asof - timedelta(days=90)).strftime("%Y%m%d")
    end = asof.strftime("%Y%m%d")

    hist = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if hist is None or hist.empty:
        return None

    if "日期" not in hist.columns or "收盘" not in hist.columns:
        return None

    hist = hist[["日期", "收盘"]].copy()
    hist["日期"] = pd.to_datetime(hist["日期"]).dt.date
    hist["收盘"] = pd.to_numeric(hist["收盘"], errors="coerce")
    hist = hist.dropna(subset=["收盘"])

    # 只取 asof 之前（严格小于）
    hist = hist[hist["日期"] < asof]
    if len(hist) < bars:
        return None

    closes = hist.tail(bars)["收盘"].tolist()
    closes.reverse()  # 转为 C[-1], C[-2], ...
    return closes


def _classify_symbol(
    symbol: str,
    open_t: float,
    close_t_1: float,
    closes: List[float],
    th: Thresholds,
) -> Optional[SignalResult]:
    # 硬过滤
    if len(closes) < 10:
        return None
    if open_t <= 0 or close_t_1 <= 0:
        return None

    c = closes
    ma5_prev = sum(c[0:5]) / 5
    ma10_prev = sum(c[0:10]) / 10
    ma5_now = (open_t + sum(c[0:4])) / 5
    ma10_now = (open_t + sum(c[0:9])) / 10

    # 防止除零
    if ma5_prev <= 0 or ma10_prev <= 0 or ma10_now <= 0:
        return None

    a5 = (ma5_now - ma5_prev) / ma5_prev
    a10 = (ma10_now - ma10_prev) / ma10_prev
    gap_prev_pct = (ma5_prev - ma10_prev) / ma10_prev
    gap_now_pct = (ma5_now - ma10_now) / ma10_now
    open_dev = abs(open_t - close_t_1) / close_t_1

    cross_up = (
        (gap_prev_pct <= 0)
        and (gap_now_pct > 0)
        and (a5 > 0)
        and (a10 > 0)
        and (a5 > a10)
        and (open_dev <= th.open_limit)
    )
    near_cross = (
        (gap_now_pct <= 0)
        and (gap_prev_pct <= 0)
        and (gap_now_pct >= -th.x)
        and (a5 > 0)
        and ((a5 - a10) >= th.y)
        and (open_dev <= th.open_limit)
    )
    hold_after_cross = (
        (a5 >= 0)
        and (a10 >= 0)
        and (a5 >= a10)
        and (gap_prev_pct > 0)
        and (gap_now_pct > 0)
        and (gap_now_pct >= th.r * gap_prev_pct)
        and (open_dev <= th.open_limit)
    )

    if cross_up:
        tier, tier_name = 1, "Cross Up"
    elif hold_after_cross:
        tier, tier_name = 3, "Hold After Cross"
    elif near_cross:
        tier, tier_name = 2, "Near Cross"
    else:
        tier, tier_name = 0, "No Signal"

    # 软标记（效果观测用）
    flags = []
    if open_dev > 0.8 * th.open_limit:
        flags.append("open_dev_near_limit")
    if abs(gap_now_pct) < 0.001:
        flags.append("ma_gap_tight")

    # 同档排序分数（可调）
    score = 1.5 * (a5 - a10) + 1.0 * gap_now_pct - 0.5 * open_dev

    return SignalResult(
        symbol=symbol,
        tier=tier,
        tier_name=tier_name,
        open_t=float(open_t),
        close_t_1=float(close_t_1),
        ma5_prev=float(ma5_prev),
        ma10_prev=float(ma10_prev),
        ma5_now=float(ma5_now),
        ma10_now=float(ma10_now),
        a5=float(a5),
        a10=float(a10),
        gap_prev_pct=float(gap_prev_pct),
        gap_now_pct=float(gap_now_pct),
        open_dev=float(open_dev),
        score=float(score),
        risk_flags=",".join(flags),
    )


def run_selector(symbols: List[str], trade_date: Optional[str], thresholds: Thresholds) -> pd.DataFrame:
    asof = _parse_trade_date(trade_date)
    today = datetime.utcnow().date()
    spot = _build_spot_snapshot().set_index("代码") if asof >= today else None

    records: List[Dict] = []
    for raw in symbols:
        symbol = str(raw).zfill(6)
        o_c = _get_open_and_prev_close(symbol=symbol, asof=asof, spot_df=spot)
        if o_c is None:
            continue
        open_t, close_t_1 = o_c

        closes = _get_recent_closes(symbol, asof, bars=10)
        if closes is None:
            continue

        result = _classify_symbol(symbol, float(open_t), float(close_t_1), closes, thresholds)
        if result is None:
            continue
        records.append(asdict(result))

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # 仅保留命中信号；按档位 + 分数降序
    df = df[df["tier"] > 0].copy()
    if df.empty:
        return df
    df = df.sort_values(["tier", "score"], ascending=[True, False]).reset_index(drop=True)
    return df


def _load_config(path: str) -> Dict[str, Any]:
    """读取 JSON 配置文件。不存在时返回空配置。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError("配置文件根节点必须是 JSON Object")
    except FileNotFoundError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="A股开盘均线三档筛选器（AKShare）")
    parser.add_argument(
        "--config",
        type=str,
        default="selector_config.json",
        help="JSON 配置文件路径，默认 selector_config.json",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="股票代码，逗号分隔，例如: 000001,600519,300750",
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="交易日，格式 YYYY-MM-DD；历史日期会使用当日开盘+前一日收盘进行回放计算",
    )
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--open-limit", type=float, default=None)
    parser.add_argument("--out", type=str, default=None, help="可选：输出 CSV 文件路径")

    args = parser.parse_args()
    config = _load_config(args.config)

    config_symbols = config.get("symbols", [])
    if isinstance(config_symbols, str):
        config_symbols = [s.strip() for s in config_symbols.split(",") if s.strip()]
    elif isinstance(config_symbols, list):
        config_symbols = [str(s).strip() for s in config_symbols if str(s).strip()]
    else:
        config_symbols = []

    cli_symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
    cli_symbols = [s for s in cli_symbols if s]
    symbols = cli_symbols if cli_symbols else config_symbols
    if not symbols:
        raise SystemExit("未提供 symbols。请通过 --symbols 或配置文件中的 symbols 提供股票代码。")

    th_conf = config.get("thresholds", {})
    if not isinstance(th_conf, dict):
        th_conf = {}
    thresholds = Thresholds(
        x=args.x if args.x is not None else float(th_conf.get("x", 0.003)),
        y=args.y if args.y is not None else float(th_conf.get("y", 0.0008)),
        r=args.r if args.r is not None else float(th_conf.get("r", 0.9)),
        open_limit=(
            args.open_limit
            if args.open_limit is not None
            else float(th_conf.get("open_limit", 0.01))
        ),
    )

    trade_date = args.trade_date if args.trade_date else config.get("trade_date")

    # 默认输出路径：output/YYYY-MM-DD.csv；--out 优先
    default_date = trade_date if trade_date else datetime.utcnow().strftime("%Y-%m-%d")
    default_out = os.path.join("output", f"{default_date}.csv")
    out = args.out if args.out else (config.get("out") or default_out)

    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = run_selector(symbols=symbols, trade_date=trade_date, thresholds=thresholds)

    if df.empty:
        print("未筛选到符合条件的股票。")
        return

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已写出结果: {out}")


if __name__ == "__main__":
    main()
