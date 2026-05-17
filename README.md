# Market-Calibrated Probabilistic Decision System

A portfolio-ready data science project that uses football betting markets as a test environment for probabilistic forecasting, market-implied probabilities, strategy selection, staking, walk-forward backtesting, and calibration evaluation.

## Goal

This is not a gambling bot.

The goal is to build a lean, end-to-end probabilistic decision system that compares model-estimated probabilities against market-implied probabilities and evaluates decision quality using historical backtesting.

## MVP Scope

- Load football match data with odds
- Clean and validate match-level data
- Calculate market-implied probabilities
- Normalize probabilities by removing bookmaker margin
- Train a baseline probability model
- Calculate edge
- Apply simple strategy rules
- Simulate flat staking
- Run walk-forward backtesting by season
- Evaluate ROI, drawdown, Brier score, and calibration
- Present saved outputs in a Streamlit dashboard

## Stack

- Python 3.11+
- uv
- pandas
- numpy
- scikit-learn
- pandera
- pyarrow
- pytest
- pyyaml
- matplotlib
- streamlit
- plotly

## Project Status

Day 1: Repo setup and environment.