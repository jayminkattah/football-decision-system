# Football Decision System

I built this project to explore a question that is more interesting than simply predicting who will win a football match:

> Can a model identify meaningful differences between its own probabilities and the probabilities implied by the betting market—and do those differences hold up when tested honestly over time?

Football betting markets are useful for this because they provide a strong, real-world benchmark. The market already absorbs a huge amount of information, so outperforming it is difficult. That makes it a much better test of a probabilistic decision system than comparing a model with a weak or artificial baseline.

This is not a gambling bot. It does not place bets, connect to bookmakers, or make claims about guaranteed profit. It is a data science project about probability, uncertainty, decision rules, and honest evaluation.

## What the project does

The pipeline:

1. Loads historical football results and decimal odds.
2. Cleans data from several source formats into one canonical match table.
3. Removes the bookmaker margin to estimate market-implied probabilities.
4. Trains a multinomial logistic-regression baseline using only pre-match information.
5. Produces predictions with walk-forward validation, training only on seasons that occurred before the season being predicted.
6. Calculates the difference between model and market probabilities.
7. Selects outcomes whose estimated edge passes a transparent threshold.
8. Simulates those decisions with a simple flat-stake policy.
9. Reports predictive performance, return, drawdown, and consistency by season.
10. Presents the saved results in a Streamlit dashboard.

The model is deliberately simple. The goal was to build the full decision and evaluation process correctly before experimenting with more complicated models.

## What I found

The current strategy does not beat the market—and I think that is an important part of the project rather than a result to hide.

Across the saved walk-forward backtest:

- 11,511 matches received out-of-sample predictions
- 9,071 opportunities passed the 2% edge threshold
- simulated ROI was approximately **-5.05%**
- none of the four evaluated seasons was profitable
- the model was slightly worse than the market on both log loss and Brier score

In other words, a model can produce probabilities that look reasonable and apparent positive edges without creating a profitable decision rule. That gap between prediction and action is the main lesson of the project.

## Project structure

```text
football-decision-system/
|-- configs/       Project configuration
|-- dashboard/     Streamlit application
|-- data/          Raw, interim, processed, and external data locations
|-- notebooks/     Inspection and analysis notebooks
|-- outputs/       Backtest reports, evaluation tables, and figures
|-- scripts/       Commands for running each pipeline stage
|-- src/fbsystem/  Reusable data, modelling, strategy, and evaluation code
|-- tests/         Automated tests
|-- pyproject.toml
`-- README.md
```

Data files and generated outputs are intentionally excluded from Git. Empty directory placeholders are included so the expected layout is still available after cloning.

## Getting started

The project requires Python 3.11 or newer. It uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <repository-url>
cd football-decision-system
uv sync
```

Place the source football CSV files under `data/external/`, then run the pipeline from the repository root:

```bash
uv run python scripts/inspect_raw_data.py
uv run python scripts/build_raw_dataset.py
uv run python scripts/build_canonical_matches.py
uv run python scripts/build_market_probabilities.py
uv run python scripts/build_predictions.py
uv run python scripts/build_strategy_opportunities.py
uv run python scripts/build_bet_simulation.py
uv run python scripts/build_backtest_summary.py
uv run python scripts/build_evaluation_outputs.py
```

Each stage reads the artifact created by the previous stage and writes a new file under `data/processed/` or `outputs/`.

## Run the dashboard

Once the pipeline artifacts have been generated:

```bash
uv run streamlit run dashboard/app.py
```

The dashboard only reads saved artifacts. It does not retrain the model or recalculate results behind the scenes.

## Run the tests

```bash
uv run pytest
```

The test suite covers data cleaning, market-probability calculations, modelling, walk-forward predictions, edge calculations, strategy selection, staking, simulation, and evaluation metrics.

## Important limitations

- The baseline largely recalibrates market probabilities; it is not yet an independent team-strength model.
- The 2% edge threshold is a simple research rule rather than an optimized production parameter.
- Historical odds do not capture every real-world execution constraint.
- Liquidity, price movement, transaction costs, account restrictions, and market impact are not modelled.
- A backtest—profitable or otherwise—is evidence about historical behavior, not a promise about the future.

## Why I built it this way

Many sports prediction projects stop after reporting accuracy. I wanted to go further and show the less glamorous parts of building a decision system: choosing a strong benchmark, preventing future leakage, translating probabilities into explicit actions, measuring downside, and being willing to report a negative result.

For me, that is the more useful data science story. The point is not that the model discovered a secret way to beat football markets. The point is that the system makes its assumptions visible and gives us an honest way to find out whether an idea works.
