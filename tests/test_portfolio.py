"""Tests for portfolio operations."""
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from portfolio import load_portfolio, save_portfolio, add_position, remove_position


def test_load_portfolio_success(tmp_path, monkeypatch):
    """Test loading valid portfolio CSV."""
    portfolio_file = tmp_path / "portfolio.csv"
    portfolio_file.write_text("symbol,shares\nAAPL,100\nTSLA,50\n")
    
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    positions = load_portfolio()
    
    assert len(positions) == 2
    assert positions[0] == {"symbol": "AAPL", "shares": "100"}
    assert positions[1] == {"symbol": "TSLA", "shares": "50"}


def test_load_portfolio_missing_file(tmp_path, monkeypatch):
    """Test loading non-existent portfolio returns empty list."""
    portfolio_file = tmp_path / "nonexistent.csv"
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    
    positions = load_portfolio()
    assert positions == []


def test_save_portfolio(tmp_path, monkeypatch):
    """Test saving portfolio to CSV."""
    portfolio_file = tmp_path / "portfolio.csv"
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    
    positions = [
        {"symbol": "AAPL", "shares": 100},
        {"symbol": "MSFT", "shares": 75}
    ]
    save_portfolio(positions)
    
    content = portfolio_file.read_text()
    assert "symbol,shares" in content
    assert "AAPL,100" in content
    assert "MSFT,75" in content


def test_add_position(tmp_path, monkeypatch):
    """Test adding new position."""
    portfolio_file = tmp_path / "portfolio.csv"
    portfolio_file.write_text("symbol,shares\nAAPL,100\n")
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    
    add_position("TSLA", 50)
    positions = load_portfolio()
    
    assert len(positions) == 2
    assert any(p["symbol"] == "TSLA" and p["shares"] == "50" for p in positions)


def test_remove_position(tmp_path, monkeypatch):
    """Test removing existing position."""
    portfolio_file = tmp_path / "portfolio.csv"
    portfolio_file.write_text("symbol,shares\nAAPL,100\nTSLA,50\n")
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    
    remove_position("AAPL")
    positions = load_portfolio()
    
    assert len(positions) == 1
    assert positions[0]["symbol"] == "TSLA"


def test_remove_nonexistent_position(tmp_path, monkeypatch):
    """Test removing position that doesn't exist."""
    portfolio_file = tmp_path / "portfolio.csv"
    portfolio_file.write_text("symbol,shares\nAAPL,100\n")
    monkeypatch.setattr("portfolio.PORTFOLIO_FILE", portfolio_file)
    
    remove_position("TSLA")  # Should not error
    positions = load_portfolio()
    
    assert len(positions) == 1  # AAPL still there
