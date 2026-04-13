# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Default run (reads selector_config.json)
python akshare_ma_selector.py

# Override symbols and/or thresholds via CLI
python akshare_ma_selector.py --symbols 000001,600519,300750 --x 0.0025

# Historical replay for a specific trade date
python akshare_ma_selector.py --trade-date 2026-01-15

# Custom output path
python akshare_ma_selector.py --out results/today.csv
```

CLI args override `selector_config.json`. Output CSVs are written to `output/YYYY-MM-DD.csv` by default.

## Architecture

Single-file project (`akshare_ma_selector.py`) with a JSON config (`selector_config.json`).

**Data flow:**
1. `main()` — parses CLI args, merges with config, calls `run_selector()`
2. `run_selector()` — iterates symbol pool; for each symbol calls `_classify_symbol()`
3. `_classify_symbol()` — fetches open/prev-close via `_get_open_and_prev_close()` and recent closes via `_get_recent_closes()`, then applies MA5/MA10 crossover logic to assign a signal tier
4. Results are filtered, sorted, and written to CSV

**Signal tiers** (defined by `Thresholds` dataclass):
- Tier 1: Cross Up — MA5 crosses above MA10 today
- Tier 2: Near Cross — approaching crossover within threshold `x`
- Tier 3: Hold After Cross — MA5 already above MA10 within threshold `y`

**Real-time vs historical:** `_get_open_and_prev_close()` uses `ak.stock_zh_a_spot_em()` for live data; when `--trade-date` is passed, it falls back to historical daily data.
