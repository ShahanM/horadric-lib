import json
from datetime import datetime
from pathlib import Path

import pandas as pd

REGISTRY_FILE = Path("data/dataset_registry.json")


def format_size(size_in_bytes: int) -> str:
    """Converts bytes to a human-readable format (MB/GB)."""
    mb = size_in_bytes / (1024 * 1024)
    if mb >= 1024:
        gb = mb / 1024
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def summarize_registry() -> None:
    if not REGISTRY_FILE.exists():
        print(
            f"Registry file not found at {REGISTRY_FILE}. Have you downloaded any datasets yet?"
        )
        return

    try:
        with open(REGISTRY_FILE, "r") as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {REGISTRY_FILE}. The file might be corrupted.")
        return

    latest = registry.get("latest", "None")
    files = registry.get("files", {})

    if not files:
        print("Registry exists but is currently empty.")
        return

    print("=" * 80)
    print(" DATASET REGISTRY SUMMARY")
    print("=" * 80)
    print(f"Currently pointing to latest: {latest}\n")

    table_data = []
    for filename, info in files.items():
        try:
            dt = datetime.fromisoformat(info.get("created_at", ""))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            date_str = "Unknown"

        size_str = format_size(info.get("size", 0))
        short_hash = info.get("hash", "")[:8] + "..." if info.get("hash") else "N/A"

        is_latest = "[*]" if filename == latest else ""

        table_data.append(
            {
                "Active": is_latest,
                "Filename": filename,
                "Created At": date_str,
                "Size": size_str,
                "SHA256": short_hash,
                "Dataset ID": info.get("dataset_id", "Unknown"),
            }
        )

    df = pd.DataFrame(table_data)
    df = df.sort_values(by="Created At", ascending=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "left")

    print(df.to_string(index=False))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    summarize_registry()
