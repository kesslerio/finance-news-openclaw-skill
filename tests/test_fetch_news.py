"""Tests for RSS feed fetching and parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from unittest.mock import Mock, patch
from fetch_news import fetch_rss, _get_best_feed_url


@pytest.fixture
def sample_rss_content():
    """Load sample RSS fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_rss.xml"
    return fixture_path.read_bytes()


def test_fetch_rss_success(sample_rss_content):
    """Test successful RSS fetch and parse."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = sample_rss_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        articles = fetch_rss("https://example.com/feed.xml")
        
        assert len(articles) == 2
        assert articles[0]["title"] == "Apple Stock Rises 5%"
        assert articles[1]["title"] == "Tesla Announces New Model"
        assert "apple-rises" in articles[0]["link"]


def test_fetch_rss_network_error():
    """Test RSS fetch handles network errors."""
    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        articles = fetch_rss("https://example.com/feed.xml")
        assert articles == []


def test_get_best_feed_url_priority():
    """Test feed URL selection prioritizes 'top' key."""
    source = {
        "name": "Test Source",
        "homepage": "https://example.com",
        "top": "https://example.com/top.xml",
        "markets": "https://example.com/markets.xml"
    }
    
    url = _get_best_feed_url(source)
    assert url == "https://example.com/top.xml"


def test_get_best_feed_url_fallback():
    """Test feed URL falls back to other http URLs when priority keys missing."""
    source = {
        "name": "Test Source",
        "homepage": "https://example.com",
        "feed": "https://example.com/feed.xml"
    }
    
    url = _get_best_feed_url(source)
    assert url == "https://example.com/feed.xml"


def test_get_best_feed_url_none_if_no_urls():
    """Test returns None when no valid URLs found."""
    source = {
        "name": "Test Source",
        "enabled": True,
        "note": "No URLs here"
    }
    
    url = _get_best_feed_url(source)
    assert url is None


def test_get_best_feed_url_skips_non_urls():
    """Test skips non-URL values."""
    source = {
        "name": "Test Source",
        "enabled": True,
        "count": 5,
        "rss": "https://example.com/rss.xml"
    }
    
    url = _get_best_feed_url(source)
    assert url == "https://example.com/rss.xml"
