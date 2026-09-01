# Market-Calibrated Probabilistic Decision System

A portfolio-ready data science project that builds a risk-aware football match decision framework using betting markets as a real-world probabilistic test environment.

This is **not a gambling bot**.

The project uses football betting markets because they provide something most toy ML projects do not: a noisy but highly informative external benchmark. The goal is not to “beat the bookies” with magic. The goal is to show how to build, test, evaluate, and communicate a disciplined probabilistic decision system under uncertainty.

---

## Project Summary

This project answers a practical decision science question:

> Given historical football results, market odds, and a baseline predictive model, can we identify situations where our model probability differs meaningfully from the market-implied probability, then evaluate those decisions honestly through walk-forward backtesting?

The system converts bookmaker odds into market-implied probabilities, trains a baseline probability model, calculates model-vs-market edge, selects opportunities through a simple strategy engine, simulates flat-stake decisions, and evaluates performance through walk-forward backtesting.

The final output is a Streamlit dashboard and a set of reproducible evaluation files.

---

## Why This Project Exists

Most beginner data science sports projects stop at predicting match outcomes.

That is fine, but incomplete.

Real decision systems need more than predictions. They need:

- calibrated probabilities
- comparison against a strong external baseline
- explicit decision rules
- capital allocation logic
- backtesting without future leakage
- honest evaluation metrics
- clear communication of limitations

This project focuses on the full decision pipeline rather than just model accuracy.

---

## Why Football Betting Markets?

Football betting odds are useful here because they act as a market-based probability benchmark.

Bookmakers and betting exchanges aggregate information from many sources, including team strength, injuries, form, public sentiment, and market activity. That makes the market difficult to beat, which is exactly why it is useful for a portfolio project.

A weak benchmark makes a project look good cheaply. A strong benchmark makes the evaluation meaningful.

The project uses the market as a test environment for probabilistic decision-making, not as encouragement to gamble.

---

## Why This Is Not a Gambling Bot

This project does **not** place bets.

It does **not** connect to bookmaker APIs.

It does **not** optimize for real-money deployment.

It does **not** recommend gambling.

Instead, it demonstrates a controlled data science workflow:

1. Convert odds into probabilities.
2. Build model probabilities.
3. Compare model and market views.
4. Apply transparent decision rules.
5. Simulate outcomes historically.
6. Evaluate results honestly.

The betting market is simply the domain used to test probability calibration, edge estimation, and decision-making under uncertainty.

---

## Repository Structure

```text
football-decision-system/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── interview_talking_points.md
│   ├── project_story.md
│   └── reproducibility.md
├── notebooks/
├── outputs/
│   ├── backtests/
│   ├── evaluation/
│   └── figures/
├── scripts/
├── src/
│   └── fbsystem/
├── tests/
├── README.md
├── pyproject.toml
└── uv.lock