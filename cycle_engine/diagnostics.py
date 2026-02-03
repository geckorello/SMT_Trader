"""Diagnostics plotting for cycle detection (optional)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .detect import detect_cycles


def plot_cycles(df: pd.DataFrame, instrument: str, out_path: str = "reports/cycle_diagnostics.png", specs=None) -> Path:
    import matplotlib.pyplot as plt

    cycles = detect_cycles(df, instrument, specs=specs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["close"], label="Close", color="#111111")

    for ctype, marker, color in [("DCL", "v", "#1f77b4"), ("HCL", "o", "#ff7f0e"), ("ICL", "^", "#2ca02c")]:
        subset = cycles[cycles["cycle_type"] == ctype]
        if not subset.empty:
            ax.scatter(subset["date"], df.set_index("date").loc[subset["date"], "close"],
                       label=ctype, marker=marker, color=color)

    ax.set_title(f"{instrument} Cycle Diagnostics")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
