import unittest
import pandas as pd
import numpy as np

from cycle_engine.detect import detect_cycles


class TestDetect(unittest.TestCase):
    def test_detect_cycle_days(self):
        # Create toy series with local lows every ~12 days and clear reversals
        pattern = [10, 9, 8, 7, 8, 9, 11, 10, 9, 8, 7, 8, 9, 11]
        close = pd.Series(pattern * 4)
        dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
        low = close - 0.5
        high = close + 0.5
        df = pd.DataFrame({"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000})

        specs = {"specs": {"GOLD": {"daily": {"min": 5, "max": 15, "unit": "days"}}}}
        cycles = detect_cycles(df, "GLD", specs=specs)
        self.assertIn("cycle_type", cycles.columns)
        self.assertIn("cycle_day", cycles.columns)
        dcl = cycles[cycles["cycle_type"] == "DCL"]
        self.assertGreaterEqual(len(dcl), 2)
        self.assertGreater(dcl.iloc[1]["cycle_day"], 1)


if __name__ == "__main__":
    unittest.main()
