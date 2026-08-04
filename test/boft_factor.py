import unittest

from lycoris.logging import logger
from lycoris.modules.boft import (
    butterfly_factor,
    log_butterfly_factor_not_found,
    log_butterfly_factor_search,
    next_higher_factor,
)


class BoftFactorTests(unittest.TestCase):
    """Tests for ``butterfly_factor``'s factor-fallback search."""

    def setUp(self):
        # The log helpers are cached, so clear them to make the assertions
        # below independent of any earlier calls in the same process.
        log_butterfly_factor_search.cache_clear()
        log_butterfly_factor_not_found.cache_clear()

    def test_valid_factor_unaffected(self):
        # 8 decomposes 128 -> (block_size, block_num).
        self.assertEqual(butterfly_factor(128, 8), (8, 16))

    def test_default_factor(self):
        # factor=-1 (auto) resolves to the full dimension.
        self.assertEqual(butterfly_factor(128), (128, 1))

    def test_oversized_factor_clamps_to_largest_valid(self):
        # 16 for 128: largest valid factor <= 16 is 16 itself.
        self.assertEqual(butterfly_factor(128, 16), (16, 8))

    def test_invalid_factor_warns_and_reports_next_higher_then_raises(self):
        # 320's valid factors are {10, 20, 40, 80, 160, 320}; 8 is too small,
        # and no lower factor works either, so this must raise ValueError.
        with self.assertLogs(logger, level="INFO") as cm:
            with self.assertRaises(ValueError) as ctx:
                butterfly_factor(320, 8)

        output = "\n".join(cm.output)
        # Warning about the search being performed.
        self.assertIn("cannot decompose dimension 320", output)
        self.assertIn("next lowest factor", output)
        self.assertIn("may not be used for all layers", output)
        # Info about the next higher factor that would work.
        self.assertIn("next higher factor that would work is 10", output)
        # The error still surfaces as before.
        self.assertIn("320", str(ctx.exception))
        self.assertIn("8", str(ctx.exception))

    def test_odd_dimension_raises_without_next_higher_hint(self):
        # Odd dimensions have no valid even block size at all, so there is no
        # higher factor to suggest.
        with self.assertLogs(logger, level="INFO") as cm:
            with self.assertRaises(ValueError):
                butterfly_factor(5, 2)

        output = "\n".join(cm.output)
        self.assertIn("No BOFT factor at or below 2 decomposes dimension 5", output)
        self.assertNotIn("next higher factor that would work", output)

    def test_downward_search_finds_lower_factor(self):
        # The real power2factorization is monotone in factor (if factor f
        # fails, every lower factor fails too), so the downward branch is not
        # reachable with real inputs. Patch it to simulate a factor that fails
        # while a lower one succeeds and verify the search logic picks it up.
        import lycoris.modules.boft as boft_mod

        real = boft_mod.power2factorization

        def fake_power2factorization(dimension, factor):
            if factor == 16:
                return None, 0
            return real(dimension, factor)

        boft_mod.power2factorization = fake_power2factorization
        try:
            with self.assertLogs(logger, level="WARNING") as cm:
                result = butterfly_factor(128, 16)
        finally:
            boft_mod.power2factorization = real

        # Falls back to the largest factor below 16 that decomposes 128 (8).
        self.assertEqual(result, (8, 16))
        self.assertIn("cannot decompose dimension 128", "\n".join(cm.output))

    def test_next_higher_factor(self):
        self.assertEqual(next_higher_factor(320, 8), 10)
        self.assertEqual(next_higher_factor(128, 2), 4)
        self.assertIsNone(next_higher_factor(5, 2))


if __name__ == "__main__":
    unittest.main()
