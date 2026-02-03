import unittest
import pandas as pd
import numpy as np

from cycle_engine import features


class TestFeatures(unittest.TestCase):
    def test_atr_basic(self):
        df = pd.DataFrame({
            "high": [10, 11, 12, 11, 13],
            "low": [9, 9, 10, 10, 11],
            "close": [9.5, 10.5, 11, 10.5, 12],
        })
        atr = features.atr(df, window=3)
        self.assertTrue(len(atr) == 5)
        self.assertTrue(np.isfinite(atr.iloc[-1]))

    def test_pivot_low(self):
        low = pd.Series([5, 4, 3, 4, 5])
        piv = features.pivot_lows(low, window=1)
        self.assertTrue(piv.iloc[2])


if __name__ == "__main__":
    unittest.main()
