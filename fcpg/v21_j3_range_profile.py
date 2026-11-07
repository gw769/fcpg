"""The two fixed, source-relative ten-command J3 diagnostic profiles."""


def range_offsets(amplitude=5):
    """Return ten negative J3 offsets; only total amplitudes 5 and 10 exist."""
    if type(amplitude) not in (int, float) or amplitude not in (5, 10):
        raise ValueError('J3 range amplitude must be exactly 5 or 10')
    if amplitude == 5:
        return (-2.5,) + (-5.,) * 9
    return (-2.5, -5., -7.5) + (-10.,) * 7


def range_result_name(amplitude=5):
    """Keep the original physical receipt separate from the amplitude-10 run."""
    range_offsets(amplitude)
    return 'j3-range-diagnostic.json' if amplitude == 5 else 'j3-range-diagnostic-10.json'
