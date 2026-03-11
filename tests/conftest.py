"""
Pytest configuration and shared fixtures.

Mocks streamlit at import time so dashboard.py can be imported
without a running Streamlit server.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Streamlit mock — must be in place before dashboard.py is imported
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
# set_page_config is called at module level in dashboard.py; make it a no-op
_st_mock.set_page_config = MagicMock()
_st_mock.cache_data = lambda **kw: (lambda f: f)   # decorator pass-through
sys.modules.setdefault("streamlit", _st_mock)
