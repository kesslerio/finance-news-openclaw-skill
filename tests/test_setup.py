"""Tests for setup wizard functionality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import json
from unittest.mock import patch
from setup import validate_language, validate_markets, create_config


def test_validate_language_valid():
    """Test valid language codes."""
    assert validate_language("en") is True
    assert validate_language("de") is True
    assert validate_language("EN") is True  # Case insensitive


def test_validate_language_invalid():
    """Test invalid language codes."""
    assert validate_language("xx") is False
    assert validate_language("") is False
    assert validate_language("english") is False


def test_validate_markets_valid():
    """Test valid market codes."""
    assert validate_markets(["US"]) is True
    assert validate_markets(["US", "EU"]) is True
    assert validate_markets(["us", "jp"]) is True  # Case insensitive


def test_validate_markets_invalid():
    """Test invalid market codes."""
    assert validate_markets([]) is False  # Empty
    assert validate_markets(["XX"]) is False  # Invalid code
    assert validate_markets(["US", "INVALID"]) is False  # Mix


def test_create_config(tmp_path, monkeypatch):
    """Test config file creation."""
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("setup.CONFIG_FILE", config_file)
    
    config_data = {
        "language": "en",
        "markets": ["US", "EU"],
        "sources": {"yahoo": {"enabled": True}}
    }
    
    create_config(config_data)
    
    assert config_file.exists()
    with open(config_file) as f:
        saved_config = json.load(f)
    
    assert saved_config["language"] == "en"
    assert saved_config["markets"] == ["US", "EU"]


def test_create_config_creates_parent_dir(tmp_path, monkeypatch):
    """Test config creation creates parent directory."""
    config_file = tmp_path / "subdir" / "config.json"
    monkeypatch.setattr("setup.CONFIG_FILE", config_file)
    
    config_data = {"language": "en"}
    create_config(config_data)
    
    assert config_file.parent.exists()
    assert config_file.exists()


@patch("builtins.input", side_effect=["en", "US,EU", "y"])
def test_setup_wizard_integration(mock_input, tmp_path, monkeypatch):
    """Test interactive setup wizard flow."""
    from setup import run_setup_wizard
    
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("setup.CONFIG_FILE", config_file)
    
    run_setup_wizard()
    
    assert config_file.exists()
    with open(config_file) as f:
        config = json.load(f)
    
    assert config["language"] == "en"
    assert "US" in config["markets"]
    assert "EU" in config["markets"]
