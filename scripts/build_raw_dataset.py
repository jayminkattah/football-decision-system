from pathlib import Path

from fbsystem.data.build_raw import (
    SELECTED_LEAGUES,
    build_raw_matches_from_inventory,
    save_raw_matches,
    select_inventory_files,
)
from fbsystem.data.external_inventory import build_external_file_inventory


EXTERNAL_DIR = Path("data/external")
RAW_OUTPUT_PATH = Path("data/raw/matches_raw.csv")
INVENTORY_OUTPUT_PATH = Path("outputs/evaluation/external_file_inventory.csv")
SELECTED_INVENTORY_OUTPUT_PATH = Path("outputs/evaluation/selected_external_files.csv")

MIN_SEASON_START_YEAR = 2020


def main() -> None:
    inventory = build_external_file_inventory(EXTERNAL_DIR)

    INVENTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_OUTPUT_PATH, index=False)

    selected_inventory = select_inventory_files(
        inventory=inventory,
        selected_leagues=SELECTED_LEAGUES,
        min_season_start_year=MIN_SEASON_START_YEAR,
    )

    selected_inventory.to_csv(SELECTED_INVENTORY_OUTPUT_PATH, index=False)

    raw_df = build_raw_matches_from_inventory(selected_inventory)
    save_raw_matches(raw_df, RAW_OUTPUT_PATH)

    print("Raw dataset build complete.")
    print(f"External files found: {len(inventory)}")
    print(f"Selected files used: {len(selected_inventory)}")
    print(f"Rows in raw dataset: {len(raw_df)}")
    print(f"Columns in raw dataset: {len(raw_df.columns)}")
    print(f"Saved inventory to: {INVENTORY_OUTPUT_PATH}")
    print(f"Saved selected files to: {SELECTED_INVENTORY_OUTPUT_PATH}")
    print(f"Saved raw dataset to: {RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()