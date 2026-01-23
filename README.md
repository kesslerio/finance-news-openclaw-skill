# Finance News Skill for Clawdbot

AI-powered market news briefings with configurable language output and automated delivery.

## Features

- 📰 **Multi-source aggregation:** WSJ, Barron's, CNBC, Yahoo Finance
- 📊 **Global markets:** US, Europe (DAX, STOXX), Japan (Nikkei)
- 🤖 **AI summaries:** Gemini-powered analysis in German or English
- 📅 **Automated briefings:** Morning (market open) and evening (market close)
- 📤 **WhatsApp delivery:** Send briefings to your group
- 📋 **Portfolio tracking:** Personalized news for your stocks

## Quick Start

```bash
# First-time setup (interactive wizard)
finance-news setup

# Generate a briefing
finance-news briefing --morning --lang de

# View market overview
finance-news market

# Get news for your portfolio
finance-news portfolio
```

## Installation

1. Clone to your Clawdbot skills directory:
   ```bash
   git clone https://github.com/kesslerio/finance-news-clawdbot-skill.git \
       ~/clawd/skills/finance-news
   ```

2. Create symlink for CLI access:
   ```bash
   ln -sf ~/clawd/skills/finance-news/scripts/finance-news ~/.local/bin/finance-news
   ```

3. Install Python dependencies:
   ```bash
   cd ~/clawd/skills/finance-news
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Run setup wizard:
   ```bash
   finance-news setup
   ```

## Configuration

The setup wizard (`finance-news setup`) lets you configure:

- **RSS Feeds:** Enable/disable news sources
- **Markets:** Choose which regions to track
- **Delivery:** WhatsApp/Telegram group for briefings
- **Language:** German or English output
- **Schedule:** Cron times for morning/evening briefings

Configuration is stored in `config/sources.json`.

## Portfolio

Manage your stock watchlist:

```bash
finance-news portfolio-list              # View portfolio
finance-news portfolio-add NVDA          # Add stock
finance-news portfolio-remove TSLA       # Remove stock
finance-news portfolio-import stocks.csv # Import from CSV
```

## Cron Jobs

The skill sets up automated briefings:

- **Morning:** 6:30 AM PT (US market open)
- **Evening:** 1:00 PM PT (US market close)

Briefings are sent to your configured WhatsApp group in German.

## Dependencies

- Python 3.10+
- feedparser (`pip install -r requirements.txt`)
- [Gemini CLI](https://github.com/google/generative-ai-cli) for AI summaries
- OpenBB (optional, for enhanced market data)
- Clawdbot for WhatsApp delivery

## License

Apache 2.0 - See [LICENSE](LICENSE) file for details.


## Related Skills

- **[task-tracker](https://github.com/kesslerio/task-tracker-clawdbot-skill):** Personal task management with daily standups, weekly reviews, and Telegram slash commands
- **oura-analytics:** Sleep and health tracking with automated reports
- **openbb:** Equity data and stock analysis (used by this skill for market quotes)
