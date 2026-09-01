import pandas as pd
import pytest

from fbsystem.staking.flat import apply_flat_stakes


def test_apply_flat_stakes_adds_constant_stake():
    bets = pd.DataFrame({"match_id": ["m1", "m2"]})

    staked = apply_flat_stakes(bets, stake=1.0)

    assert staked["stake"].tolist() == [1.0, 1.0]
    assert "stake" not in bets.columns


def test_apply_flat_stakes_rejects_non_positive_stake():
    with pytest.raises(ValueError, match="stake"):
        apply_flat_stakes(pd.DataFrame({"match_id": ["m1"]}), stake=0)
