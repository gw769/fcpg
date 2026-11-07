"""Shared pytest configuration for the test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running tests (e.g. real MuJoCo rendering); deselect "
        "with -m 'not slow'",
    )
