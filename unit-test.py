import unittest
import logging
import coverage

cov = coverage.Coverage()
cov.start()

from lycoris.logging import logger

logger.setLevel(logging.ERROR)

from test.module import LycorisModuleTests
from test.wrapper import LycorisWrapperTests
from test.functional import LycorisFunctionalTests
from test.weight_decompose import WeightDecomposeTests


TESTS = [
    LycorisModuleTests,
    LycorisFunctionalTests,
    LycorisWrapperTests,
    WeightDecomposeTests,
]

try:
    from test.kohya import LycorisKohyaWrapperTests
    TESTS.append(LycorisKohyaWrapperTests)
except Exception:
    print("Skipping Kohya tests (Kohya SD-Scripts not available)")


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=0)
    for test in TESTS:
        suite = test_loader.loadTestsFromTestCase(test)
        result = runner.run(suite)

    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")
