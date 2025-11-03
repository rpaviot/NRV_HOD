"""
Shared utility functions for test scripts.

This module provides common formatting and timing utilities used across
multiple test files in the HOD_NRV test suite.
"""


def print_header(title: str) -> None:
    """
    Print formatted section header.

    Parameters
    ----------
    title : str
        Header title text

    Examples
    --------
    >>> print_header("Step 1: Load Data")

    ======================================================================
      Step 1: Load Data
    ======================================================================
    """
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_timing(step_name: str, elapsed_time: float, units: str = "s") -> None:
    """
    Print timing result in a formatted way.

    Parameters
    ----------
    step_name : str
        Name of the timed operation
    elapsed_time : float
        Elapsed time in seconds (or milliseconds if units='ms')
    units : str, default='s'
        Time units: 's' for seconds, 'ms' for milliseconds

    Examples
    --------
    >>> import time
    >>> start = time.time()
    >>> # ... do some work ...
    >>> elapsed = time.time() - start
    >>> print_timing("Data loading", elapsed)
      ⏱  Data loading                              1.234 s

    >>> print_timing("Quick operation", elapsed * 1000, units='ms')
      ⏱  Quick operation                           123.40 ms
    """
    if units == "ms":
        print(f"  ⏱  {step_name:40s} {elapsed_time*1000:8.2f} ms")
    else:
        print(f"  ⏱  {step_name:40s} {elapsed_time:8.3f} s")
