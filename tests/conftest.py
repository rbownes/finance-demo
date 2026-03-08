"""
Shared fixtures and setup for the finance-demo test suite.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Streamlit stub
# ---------------------------------------------------------------------------
# dashboard.py calls st.set_page_config() and uses @st.cache_data at module
# level. Mock the entire streamlit namespace before any test file imports
# dashboard so that the top-level side-effects are no-ops.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()

# cache_data / cache_resource must return a decorator that wraps the function
# unchanged so the underlying logic remains testable.
def _passthrough_decorator(*_args, **_kwargs):
    def _wrap(fn):
        return fn
    return _wrap

_st_mock.cache_data.side_effect = _passthrough_decorator
_st_mock.cache_resource.side_effect = _passthrough_decorator

sys.modules.setdefault("streamlit", _st_mock)
