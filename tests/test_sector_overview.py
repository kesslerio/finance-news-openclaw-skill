"""Tests for the weekly sector overview report."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import sector_overview


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 2, 12)


class FakeCloses:
    def __init__(self, values):
        self.values = values
        self.iloc = self

    def dropna(self):
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class FakeQuoteFrame:
    columns = ["Close"]

    def __init__(self, closes):
        self.closes = FakeCloses(closes)

    def __getitem__(self, column):
        assert column == "Close"
        return self.closes


class FakeHistory:
    class Columns:
        nlevels = 2

    columns = Columns()

    def __init__(self, closes_by_symbol):
        self.closes_by_symbol = closes_by_symbol
        self.empty = not bool(closes_by_symbol)

    def xs(self, symbol, level, axis, drop_level):
        assert level in (0, 1)
        assert axis == 1
        assert drop_level is True
        if symbol not in self.closes_by_symbol:
            raise KeyError(symbol)
        return FakeQuoteFrame(self.closes_by_symbol[symbol])


def test_format_sector_overview_groups_sectors_by_weekly_performance(monkeypatch):
    changes = {
        "XLK": 4.2,
        "XLC": 3.1,
        "XLY": -1.0,
        "XLE": -3.3,
        "XLF": 0.4,
        "XLI": 0.1,
        "XLU": -2.2,
        "XLB": -1.8,
        "XLRE": 0.0,
        "XLV": 2.0,
        "XLP": 1.0,
    }
    monkeypatch.setattr(sector_overview, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        sector_overview,
        "get_weekly_market_data",
        lambda: {
            **changes,
            "^GSPC": 1.25,
        },
    )

    report = sector_overview.format_sector_overview()

    assert report.startswith("📊 **Sektor-Übersicht** (Woche 05.02-12.02)")
    assert "S&P 500: +1.25%" in report
    assert report.index("Technology (XLK): +4.20%") < report.index("Communication Services (XLC): +3.10%")
    assert report.index("Energy (XLE): -3.30%") < report.index("Utilities (XLU): -2.20%")
    assert report.index("Utilities (XLU): -2.20%") < report.index("Materials (XLB): -1.80%")
    assert "⚪ **Mittelfeld:**" in report
    assert report.index("Consumer Staples (XLP): +1.00%") < report.index("Financials (XLF): +0.40%")
    for symbol in changes:
        assert report.count(f"({symbol})") == 1


def test_format_sector_overview_reports_when_no_sector_data_is_available(monkeypatch):
    monkeypatch.setattr(sector_overview, "get_weekly_market_data", lambda: {})

    assert sector_overview.format_sector_overview() == "⚠️ No sector data available"


def test_weekly_market_data_calculates_five_session_performance_in_one_batch(monkeypatch):
    history = FakeHistory(
        {
            "XLK": [100, 101, 102, 103, 104, 105],
            "XLC": [45, 45.5, 46, 46.5, 47, 48],
            "^GSPC": [400, 402, 403, 404, 405, 408],
        }
    )
    download = Mock(return_value=history)
    monkeypatch.setattr(sector_overview, "_download_history", download)
    data = sector_overview.get_weekly_market_data()

    assert data["XLK"] == 5.0
    assert abs(data["XLC"] - 6.6667) < 0.0001
    assert data["^GSPC"] == 2.0
    assert download.call_count == 1
    assert download.call_args.args[0] == [*sector_overview.SECTOR_ETFS, "^GSPC"]


def test_format_sector_overview_does_not_repeat_partial_sector_data(monkeypatch):
    changes = {"XLK": 4.0, "XLC": 3.0, "XLY": 2.0, "XLE": -3.0}
    monkeypatch.setattr(sector_overview, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        sector_overview,
        "get_weekly_market_data",
        lambda: changes,
    )

    report = sector_overview.format_sector_overview()

    assert "Energy (XLE): -3.00%" in report
    assert "🔴 **Schwächste Performer:**" in report
    assert "⚪ **Mittelfeld:**" not in report
    for symbol in changes:
        assert report.count(f"({symbol})") == 1


def test_send_to_whatsapp_uses_openclaw_and_finance_news_target(monkeypatch):
    monkeypatch.setenv("FINANCE_NEWS_TARGET", "weekly-market-group")
    run = Mock(return_value=Mock(returncode=0, stderr=""))
    monkeypatch.setattr(sector_overview.subprocess, "run", run)

    assert sector_overview.send_to_whatsapp("weekly report") is True

    assert run.call_args.args[0] == [
        "openclaw",
        "message",
        "send",
        "--channel",
        "whatsapp",
        "--target",
        "weekly-market-group",
        "--message",
        "weekly report",
    ]
    assert run.call_args.kwargs == {"capture_output": True, "text": True, "timeout": 30}


def test_send_to_whatsapp_requires_a_target(monkeypatch):
    monkeypatch.delenv("FINANCE_NEWS_TARGET", raising=False)
    run = Mock()
    monkeypatch.setattr(sector_overview.subprocess, "run", run)

    assert sector_overview.send_to_whatsapp("weekly report") is False
    run.assert_not_called()
