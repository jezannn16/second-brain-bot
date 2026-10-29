import os

import pytest

from second_brain import fetch_url


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to use the network")
def test_fetch_public_example_page():
    source, content = fetch_url("https://example.com", timeout=12)

    assert source == "https://example.com"
    assert "Example Domain" in content
