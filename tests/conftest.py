"""pytest configuration: ensure the src/ layout is importable."""
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Forecast-origin sample window (Objective B): the canonical production window
# is [2022-01-01, 2023-01-01) UTC by forecast origin. Most synthetic model
# fixtures use out-of-window dates (e.g. 2024) and only exercise pipeline
# mechanics, so ROW FILTERING is DISABLED by default for the test session
# (forecast_origin is still stamped on every signal frame). Tests that verify
# the real boundary behaviour opt in explicitly by passing a config to
# ``restrict_to_forecast_sample`` or by setting these env vars themselves.
os.environ.setdefault("THESIS_FORECAST_SAMPLE_ENABLED", "0")
