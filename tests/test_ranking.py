import sys
from pathlib import Path
import pytest
from datetime import datetime, timedelta

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ranking import calculate_score, rank_headlines, classify_category

def test_classify_category():
    assert "macro" in classify_category("Fed signals rate cut")
    assert "company_specific" in classify_category("Apple earnings beat")
    assert "equity_broad" in classify_category("S&P 500 tech sector earnings beat expectations")
    assert "energy" in classify_category("Oil prices surge")
    assert "tech" in classify_category("AI chip demand remains high")
    assert "geopolitics" in classify_category("US imposes new sanctions on Russia")
    assert classify_category("Weather is nice") == ["general"]


def test_classify_category_downgrade_not_broad_context():
    categories = classify_category("Apple downgrade follows weak iPhone demand")
    assert "company_specific" in categories
    assert "equity_broad" not in categories

def test_calculate_score_impact():
    weights = {"market_impact": 0.4, "novelty": 0.2, "breadth": 0.2, "credibility": 0.1, "diversity": 0.1}
    category_counts = {}
    
    high_impact = {"title": "Fed announces emergency rate cut", "source": "Reuters", "published_at": datetime.now().isoformat()}
    low_impact = {"title": "Local coffee shop opens", "source": "Blog", "published_at": datetime.now().isoformat()}
    
    score_high = calculate_score(high_impact, weights, category_counts)
    score_low = calculate_score(low_impact, weights, category_counts)
    
    assert score_high > score_low

def test_rank_headlines_deduplication():
    headlines = [
        {"title": "Fed signals rate cut in March", "source": "WSJ"},
        {"title": "FED SIGNALS RATE CUT IN MARCH!!!", "source": "Reuters"}, # Dupe
        {"title": "Apple earnings are out", "source": "CNBC"}
    ]
    
    result = rank_headlines(headlines)
    
    # After dedupe, we should have 2 unique headlines
    assert result["after_dedupe"] == 2
    # must_read should contain the best ones
    assert len(result["must_read"]) <= 2

def test_rank_headlines_sorting():
    headlines = [
        {"title": "Local news", "source": "SmallBlog", "description": "Nothing much"},
        {"title": "FED EMERGENCY RATE CUT", "source": "Bloomberg", "description": "Huge market impact"},
        {"title": "Nvidia Earnings Surprise", "source": "Reuters", "description": "AI demand surges"}
    ]
    
    result = rank_headlines(headlines)
    
    # FED should be first due to macro impact + credibility
    assert "FED" in result["must_read"][0]["title"]
    assert any("Nvidia" in item["title"] for item in result["must_read"])

def test_source_cap():
    # Test that we don't have too many items from the same source
    headlines = [
        {"title": f"Story {i}", "source": "Reuters"} for i in range(10)
    ]
    
    # Default source cap is 2
    result = rank_headlines(headlines)
    
    reuters_in_must_read = [h for h in result["must_read"] if h["source"] == "Reuters"]
    reuters_in_scan = [h for h in result["scan"] if h["source"] == "Reuters"]
    
    assert len(reuters_in_must_read) + len(reuters_in_scan) <= 2


def test_company_specific_earnings_penalty_vs_broad_context():
    weights = {"market_impact": 0.4, "novelty": 0.2, "breadth": 0.2, "credibility": 0.1, "diversity": 0.1}
    category_counts = {}
    company_story = {
        "title": "Apple earnings beat estimates",
        "description": "Company reports strong EPS guidance",
        "source": "Reuters",
        "published_at": datetime.now().isoformat(),
    }
    broad_story = {
        "title": "S&P 500 tech sector earnings beat expectations",
        "description": "Broad market index gains on strong sector results",
        "source": "Reuters",
        "published_at": datetime.now().isoformat(),
    }

    company_score = calculate_score(company_story, weights, category_counts)
    broad_score = calculate_score(broad_story, weights, category_counts)
    assert broad_score > company_score


def test_diversity_does_not_force_company_specific():
    headlines = [
        {"title": "Fed signals rate cut path", "source": "Reuters"},
        {"title": "US sanctions tighten on Russia exports", "source": "Bloomberg"},
        {"title": "S&P 500 tech sector earnings trend improves", "source": "WSJ"},
        {"title": "Oil rises as OPEC output drops", "source": "FT"},
        {"title": "Apple earnings beat estimates", "source": "CNBC"},
    ]
    result = rank_headlines(headlines)
    top_titles = [item["title"] for item in result["must_read"][:3]]
    assert "Apple earnings beat estimates" not in top_titles
