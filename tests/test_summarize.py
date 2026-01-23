"""Tests for summarize helpers."""

from datetime import datetime

import summarize


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 15, 0)


def test_generate_briefing_auto_time_evening(capsys, monkeypatch):
    def fake_market_news(*_args, **_kwargs):
        return {
            "headlines": [
                {"source": "CNBC", "title": "Headline one"},
                {"source": "Yahoo", "title": "Headline two"},
                {"source": "CNBC", "title": "Headline three"},
            ],
            "markets": {
                "us": {
                    "name": "US Markets",
                    "indices": {
                        "^GSPC": {"name": "S&P 500", "data": {"price": 100, "change_percent": 1.0}},
                    },
                }
            },
        }

    def fake_summary(*_args, **_kwargs):
        return "OK"

    monkeypatch.setattr(summarize, "get_market_news", fake_market_news)
    monkeypatch.setattr(summarize, "get_portfolio_news", lambda *_a, **_k: None)
    monkeypatch.setattr(summarize, "summarize_with_claude", fake_summary)
    monkeypatch.setattr(summarize, "datetime", FixedDateTime)

    args = type(
        "Args",
        (),
        {
            "lang": "de",
            "style": "briefing",
            "time": None,
            "model": "claude",
            "json": False,
            "research": False,
        },
    )()

    summarize.generate_briefing(args)
    stdout = capsys.readouterr().out
    assert "Börsen-Abend-Briefing" in stdout
