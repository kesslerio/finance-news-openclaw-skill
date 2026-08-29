import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import translate_portfolio
from translate_portfolio import has_pretranslated_portfolio


PROJECT_ROOT = Path(__file__).parent.parent


def test_has_pretranslated_portfolio_true_when_title_de_present():
    data = {
        "raw_data": {
            "portfolio": {
                "stocks": {
                    "NOVO-B.CO": {
                        "articles": [
                            {"title": "Title", "title_de": "Titel"},
                            {"title": "Another", "title_de": "Noch einer"},
                        ]
                    }
                }
            }
        }
    }
    assert has_pretranslated_portfolio(data) is True


def test_has_pretranslated_portfolio_false_when_missing_title_de():
    data = {
        "raw_data": {
            "portfolio": {
                "stocks": {
                    "AAPL": {
                        "articles": [
                            {"title": "Apple updates guidance"}
                        ]
                    }
                }
            }
        }
    }
    assert has_pretranslated_portfolio(data) is False


def test_has_pretranslated_portfolio_false_when_partial_translation():
    data = {
        "raw_data": {
            "portfolio": {
                "stocks": {
                    "NOVO-B.CO": {
                        "articles": [
                            {"title": "Title one", "title_de": "Titel eins"},
                            {"title": "Title two"},
                        ]
                    }
                }
            }
        }
    }
    assert has_pretranslated_portfolio(data) is False


def test_translate_headlines_calls_ornith_directly(monkeypatch):
    captured = {}

    def fake_call(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return '["Titel"]'

    monkeypatch.setenv("KALLIOPE_SERVING_API_KEY", "test-key")
    monkeypatch.setattr(translate_portfolio, "call_openai_chat", fake_call)

    assert translate_portfolio.translate_headlines(["Title"]) == ["Titel"]
    assert captured["model"] == "ornith-1.5:35b-medium"
    assert captured["api_key"] == "test-key"


def test_translate_headlines_rejects_non_ornith_override(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-Ornith override must fail before network I/O")

    monkeypatch.setenv("KALLIOPE_SERVING_API_KEY", "test-key")
    monkeypatch.setenv("FINANCE_NEWS_ORNITH_MODEL", "qwen3.8:27b-fast")
    monkeypatch.setattr(translate_portfolio, "call_openai_chat", fail_if_called)

    assert translate_portfolio.translate_headlines(["Title"]) == ["Title"]


def test_scheduled_workflows_expose_only_ornith():
    for name in ("briefing.yaml", "briefing-cron.yaml"):
        workflow = (PROJECT_ROOT / "workflows" / name).read_text()
        assert "FINANCE_NEWS_DS4" not in workflow
        assert "qwen|ds4" not in workflow
        assert "ornith) ;;" in workflow


def test_portfolio_translation_does_not_attach_to_openclaw_agent():
    source = (PROJECT_ROOT / "scripts" / "translate_portfolio.py").read_text()
    assert "openclaw" not in source.lower()
    assert "subprocess.run" not in source
