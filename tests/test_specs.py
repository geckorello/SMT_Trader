import unittest

from cycle_engine.pdf_specs import parse_specs_from_text


class TestSpecs(unittest.TestCase):
    def test_parse_specs(self):
        text = "Gold daily cycle count: Day 22 (average duration 30-40 days)"
        specs = parse_specs_from_text(text, "test.pdf")
        self.assertIn("GOLD", specs)
        self.assertIn("daily", specs["GOLD"])
        self.assertEqual(specs["GOLD"]["daily"]["min"], 30)


if __name__ == "__main__":
    unittest.main()
