"""Setup for pytest."""

import pendulum


def timestamper() -> str:
    """Return formatted timestamp for log messages."""
    return f"{pendulum.now().strftime('%F %T.%f')} |>  "
