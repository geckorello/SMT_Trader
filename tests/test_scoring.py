import unittest

from cycle_engine.scoring import score_cycle


class TestScoring(unittest.TestCase):
    def test_score_range(self):
        score, notes, comp = score_cycle(
            pivot_strength=0.5,
            reversal=True,
            drawdown=-0.1,
            atr_pattern=1.0,
            length=30,
            expected=(20, 40),
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("length=30", notes)


if __name__ == "__main__":
    unittest.main()
